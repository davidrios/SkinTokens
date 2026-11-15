import argparse
import contextlib
import os
import secrets
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"


def _install_lightning_stub() -> None:
    """Register a minimal stand-in for ``lightning`` before ``src`` imports it.

    Inference only uses Lightning's base classes, but importing it costs ~3.3s:
    it imports all of torchmetrics (text/image/audio metrics, torchvision,
    scipy.signal) just to read its version. The stand-in provides only what the
    inference path touches; set SKINTOKENS_REAL_LIGHTNING=1 to use the real one.
    """
    import sys
    import types
    from typing import Any

    class LightningModule(torch.nn.Module):
        _trainer = None
        current_epoch = 0
        global_step = 0

        @property
        def trainer(self):
            raise RuntimeError("demo.py runs without a Lightning Trainer")

        @property
        def device(self) -> torch.device:
            param = next(self.parameters(), None)
            return param.device if param is not None else torch.device("cpu")

        def save_hyperparameters(self, *args, **kwargs) -> None:
            pass

        def on_load_checkpoint(self, checkpoint) -> None:
            pass

    class LightningDataModule:
        def __init__(self, *args, **kwargs) -> None:
            pass

    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    utilities = types.ModuleType("lightning.pytorch.utilities")
    types_mod = types.ModuleType("lightning.pytorch.utilities.types")
    pytorch.LightningModule = LightningModule
    pytorch.LightningDataModule = LightningDataModule
    types_mod.EVAL_DATALOADERS = types_mod.TRAIN_DATALOADERS = Any
    lightning.pytorch, pytorch.utilities, utilities.types = pytorch, utilities, types_mod
    sys.modules.update({
        "lightning": lightning,
        "lightning.pytorch": pytorch,
        "lightning.pytorch.utilities": utilities,
        "lightning.pytorch.utilities.types": types_mod,
    })


if os.environ.get("SKINTOKENS_REAL_LIGHTNING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
    _install_lightning_stub()
    # transformers imports scikit-learn (~0.6s) only for assisted generation,
    # which greedy decoding never uses; a None entry makes it look absent.
    import sys
    sys.modules.setdefault("sklearn", None)

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.seeding import derive_seed, enable_deterministic_kernels, sample_point_clouds, seeded_rng
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import get_model
from src.paths import EXPERIMENTS_DIR
from src.rig_package.parser.glb import transfer_rigging
from src.model import decode_attention
from src.model.grammar import enable_cuda_deferred_stop
from src.server import config as server_config
from src.server.pipeline import _get_fast_decode_compile_config, _set_llm_attention

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_CKPTS = [
    str(EXPERIMENTS_DIR / "articulation_xl_quantization_256_token_4/grpo_1400.ckpt"),
]

model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None


@contextlib.contextmanager
def skip_weight_init():
    """Make torch.nn.init a no-op while the model is built.

    Every weight is overwritten by the checkpoint right after (loaded with
    strict=True, so a weight missing from it is an error, never left random),
    which makes the random init (~1.2s) wasted work.
    """
    import torch.nn.init as init
    names = [n for n in dir(init) if n.endswith("_") and not n.startswith("_")
             and callable(getattr(init, n))]
    saved = {n: getattr(init, n) for n in names}
    for n in names:
        setattr(init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(init, n, fn)


def load_model(model_ckpt: str, hf_path: Optional[str], device: Optional[str] = None) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
    if model is not None and model_ckpt == CURRENT_MODEL_CKPT and hf_path == CURRENT_HF_PATH:
        return ("Model already loaded.", model_ckpt)

    if not model_ckpt:
        raise RuntimeError("model_ckpt is empty. Please select a checkpoint.")

    print(f"Loading model: {model_ckpt}, hf_path={hf_path}")
    started_at = time.perf_counter()
    with skip_weight_init():
        model = get_model(model_ckpt, hf_path=hf_path, device=device)
    print(f"[PROFILE] Model load: {time.perf_counter() - started_at:.2f}s")
    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    CURRENT_MODEL_CKPT = model_ckpt
    CURRENT_HF_PATH = hf_path
    return ("Model loaded.", model_ckpt)


SUPPORTED_EXT = {".glb", ".gltf"}


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def map_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    rel = in_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".glb")


def sample_output_path(out_path: Path, sample_idx: int, num_samples: int) -> Path:
    if num_samples <= 1:
        return out_path
    return out_path.with_name(f"{out_path.stem}_{sample_idx}{out_path.suffix}")


def run_rig(
    filepaths: List[Path],
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
    seed: int,
    num_samples: int = 1,
    auto_ground: bool = True,
):
    """Rig each file with greedy decoding: ``num_samples`` rigs per file, each
    from its own point sampling of the mesh, all reproducible from ``seed``."""
    assert len(filepaths) == len(output_paths)

    # Off by default (same as the server): a seed then reproduces only within
    # this process. Set SKINTOKENS_DETERMINISTIC=1 for cross-run reproducibility
    # at the cost of ~25% decode speed.
    if os.environ.get("SKINTOKENS_DETERMINISTIC", "0").strip().lower() in {"1", "true", "yes", "on"}:
        enable_deterministic_kernels()
    # Same decode setup as the server (src/server/pipeline.py): generate() checks
    # for its end one step late instead of syncing every token, and the fast path
    # decodes over a static KV cache with CUDA-graph replay.
    decode_attention.register()
    enable_cuda_deferred_stop()
    load_model(model_ckpt, hf_path)

    generate_kwargs = dict(
        max_new_tokens=server_config.MAX_NEW_TOKENS,
        use_cache=True,
    )
    fast_decode = server_config.FAST_DECODE and torch.device(model.device).type == "cuda"
    if fast_decode:
        try:
            generate_kwargs.update(
                cache_implementation="static",
                compile_config=_get_fast_decode_compile_config(),
            )
            _set_llm_attention(model, decode_attention.DECODE_ATTN_IMPL)
        except Exception as exc:
            print(f"[fast-decode] unavailable ({exc}); using the default path")
            fast_decode = False
    print(f"[fast-decode] {'on' if fast_decode else 'off'}"
          + (f" ({decode_attention.DECODE_ATTN_MODE} attention, {server_config.FAST_DECODE_BACKEND})"
             if fast_decode else ""))

    datapath = {
        "data_name": None,
        "loader": "glb",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataset = module.predict_dataloader()["articulation"].dataset

    results_out = []

    def synchronize_inference_device():
        device = model.device
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    for i in tqdm(range(len(dataset))):
        # Every file is seeded on its own, so a file's result does not depend
        # on which other files are in the run.
        with seeded_rng(seed):
            batch = dataset.collate_fn([dataset[i]])
        asset = batch["model_input"][0].asset
        vertices, normals = sample_point_clouds(asset, transform.sampler, seed, num_samples)
        batch["vertices"], batch["normals"] = vertices.unsqueeze(0), normals.unsqueeze(0)
        batch = {
            k: v.to(model.device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        # Inputs are plain meshes: the model always generates skeleton + skin.
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)

        # Decoding is fixed to greedy single-beam search.
        batch["generate_kwargs"] = dict(generate_kwargs)

        # Synchronize around the timed section so CUDA work launched
        # asynchronously is included in the measured inference duration.
        # Batch loading/GLB parsing and GLB export both happen outside it.
        synchronize_inference_device()
        inference_started_at = time.perf_counter()
        with seeded_rng(seed), torch.inference_mode():
            preds: List[TokenRigResult] = model.predict_step(
                batch,
                make_asset=True,
            )["results"]
        synchronize_inference_device()
        inference_elapsed = time.perf_counter() - inference_started_at
        print(f"[PROFILE] Inference for {filepaths[i].name}: {inference_elapsed:.3f}s")

        for sample_idx, pred in enumerate(preds):
            asset = pred.asset
            assert asset is not None
            collapsed_joints = asset.collapse_near_parent_joints()
            if collapsed_joints:
                print(f"[postprocess] Collapsed near-parent joints: {', '.join(collapsed_joints)}")

            out_path = sample_output_path(output_paths[i], sample_idx, len(preds))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with seeded_rng(derive_seed(seed, sample_idx)):
                transfer_rigging(
                    source_asset=asset,
                    target_path=asset.path,
                    export_path=str(out_path),
                    group_per_vertex=4,
                    auto_ground=auto_ground,
                )

            print(f"[OK] Exported: {out_path}")

            results_out.append(out_path)

    return results_out


def run_cli(args):
    input_path = Path(args.input).resolve()
    
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = (Path("outputs") / f"{input_path.stem}_bind.glb").resolve()

    files = collect_files(input_path)
    if not files:
        raise RuntimeError("No valid 3D files found.")

    if len(files) == 1 and output_path.suffix:
        outputs = [output_path]
    else:
        outputs = [
            map_output_path(f, input_path, output_path)
            for f in files
        ]

    seed = args.seed if args.seed is not None else secrets.randbits(63)
    if args.seed is None:
        print(f"[seed] {seed} (pass --seed to reproduce)")
    run_rig(
        files,
        outputs,
        args.model_ckpt,
        args.hf_path,
        seed=seed,
        num_samples=args.num_samples,
        auto_ground=args.auto_ground,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", default=None, help="Output file or directory (default: outputs/{input_name}_bind.glb)")

    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=1,
                        help="Rigs per input, each from a different point sampling of the mesh (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the point sampling; the same seed and --num-samples reproduce "
                             "the same outputs (default: random, printed)")

    parser.add_argument("--auto-ground", action=argparse.BooleanOptionalAction, default=True,
                        help="Move the rigged model so its bottom sits on Z=0 "
                             "(default: on; --no-auto-ground keeps it in place)")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required for CLI mode. Use serve.py for the HTTP API server.")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")

    run_cli(args)
