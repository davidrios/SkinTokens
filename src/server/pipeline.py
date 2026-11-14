"""The model pipeline as a worker process runs it.

Each worker process (:mod:`src.server.worker`) loads its own copy of the model
with :func:`load` and serves one request at a time with :func:`run_job`:
parse the upload, sample the rig on the GPU, export the GLB(s). Nothing here is
shared between requests running in different workers, so the stages need no
locks: the parallelism comes from running several workers.

.. note::

    Results are **non-deterministic** — the same input file may produce a
    different skeleton (joint count / structure) across runs, even with identical
    parameters (``do_sample=False``, same ``num_beams``).  This is inherent to
    flash_attention_2 + bf16: the tiling algorithm introduces tiny floating-point
    differences per forward pass, which accumulate over 28 transformer layers and
    cause beam‑search tie‑breaking to diverge.  TF32 remains enabled for speed;
    disabling it would reduce — but not eliminate — this effect.
"""
from __future__ import annotations

import os

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
# expandable_segments needs CUDA VMM APIs, which WSL2 (/dev/dxg) lacks; recent
# torch then fails every allocation with "CUDA driver error: unknown error".
if not os.path.exists("/dev/dxg"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import dataclasses
import gc
import logging
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import torch

from src.server import config
from src.server.config import GenParams
from src.server.worker import Cancellation, GenerationCancelled, ServiceSpec

logger = logging.getLogger("skintokens.serve")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = os.environ.get("SKINTOKENS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from src.model.attention import LLM_ATTN_IMPL as _DEFAULT_ATTN_IMPL, SDPA_ATTN_IMPL as _SDPA_ATTN_IMPL

__all__ = ["GenerationCancelled", "load", "run_job", "SERVICE"]


class _State:
    model = None
    tokenizer = None
    transform = None


state = _State()

# Built lazily so a transformers build without CompileConfig disables fast decode
# gracefully rather than failing at import time.
_fast_decode_compile_config = None


def _set_llm_attention(model, impl: str) -> None:
    """Switch only the LLM transformer's attention backend.

    The mesh encoder and VAE call ``flash_attn_func`` directly (not via config),
    so they are unaffected; only Qwen3's config-dispatched attention changes.
    Flash-attention mis-masks the pre-allocated static cache (it attends over the
    unfilled slots), so the fast path must run SDPA. No-op if already set.
    """
    cfg = getattr(model.transformer, "config", None)
    if cfg is not None and getattr(cfg, "_attn_implementation", None) == impl:
        return
    for module in model.transformer.modules():
        mcfg = getattr(module, "config", None)
        if mcfg is not None and hasattr(mcfg, "_attn_implementation"):
            mcfg._attn_implementation = impl


def _get_fast_decode_compile_config():
    """Return a cached CompileConfig for the configured fast-decode backend."""
    global _fast_decode_compile_config
    if _fast_decode_compile_config is None:
        from transformers import CompileConfig
        if config.FAST_DECODE_BACKEND == "inductor":
            # Inductor fuses pointwise ops; "reduce-overhead" layers CUDA graphs on
            # top. Requires Triton. fullgraph stays False because data-dependent ops
            # in the attention path introduce graph breaks.
            _fast_decode_compile_config = CompileConfig(
                backend="inductor", mode="reduce-overhead", fullgraph=False
            )
        else:
            # Triton-free CUDA-graph capture (no fusion). The ``cudagraphs`` backend
            # rejects the ``mode`` argument, so it must be left unset.
            _fast_decode_compile_config = CompileConfig(
                backend="cudagraphs", mode=None, fullgraph=False
            )
    return _fast_decode_compile_config


def run_startup_stage(label: str, operation):
    """Run a startup operation with start/end logs and an elapsed heartbeat."""
    started_at = time.monotonic()
    finished = threading.Event()
    logger.info("Startup stage started: %s", label)

    def heartbeat() -> None:
        while not finished.wait(config.STARTUP_HEARTBEAT_SEC):
            logger.info("Startup stage in progress: %s elapsed=%.2fs",
                        label, time.monotonic() - started_at)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = operation()
    except Exception:
        logger.exception("Startup stage failed: %s elapsed=%.2fs",
                         label, time.monotonic() - started_at)
        raise
    finally:
        finished.set()
        heartbeat_thread.join()
    logger.info("Startup stage completed: %s elapsed=%.2fs",
                label, time.monotonic() - started_at)
    return result


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _prewarm_fast_decode() -> None:
    """Trigger the fast-decode compile + CUDA-graph capture on a synthetic input.

    The transformer's prefill/decode graph shapes are independent of the input
    geometry (the mesh encoder emits a fixed number of condition tokens and the
    no-skeleton prompt is just ``[bos, cls]``), so a random point cloud warms the
    exact graphs real requests will replay. Must run on the thread that later
    serves requests (the worker's main thread) so the graph is captured there.
    Uses the real ``MAX_NEW_TOKENS`` so the static-cache shape matches, but stops
    after a handful of tokens (the graph is captured within the first few steps)
    and skips VAE decode via ``only_ids`` since the truncated sequence isn't a rig.
    """
    model = state.model
    if model is None:
        return
    try:
        compile_config = _get_fast_decode_compile_config()
    except Exception as exc:
        logger.warning("Fast decode prewarm skipped (CompileConfig unavailable): %s", exc)
        return

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAfter(StoppingCriteria):
        def __init__(self, k: int) -> None:
            self.k = k

        def __call__(self, input_ids, scores, **kwargs):
            done = bool(input_ids.shape[1] >= self.k)
            return torch.full(
                (input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device
            )

    _set_llm_attention(model, _SDPA_ATTN_IMPL)
    device = model.device
    n_pts = 8192
    vertices = torch.rand(n_pts, 3, device=device) * 2.0 - 1.0
    normals = torch.nn.functional.normalize(torch.randn(n_pts, 3, device=device), dim=-1)
    batch = {
        "vertices": vertices,
        "normals": normals,
        "cls": ["prewarm"],
        "generate_kwargs": dict(
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_beams=1,
            num_samples=1,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=1.0,
            use_cache=config.USE_GENERATION_CACHE,
            only_ids=True,
            cache_implementation="static",
            compile_config=compile_config,
            stopping_criteria=StoppingCriteriaList([_StopAfter(32)]),
        ),
    }
    # Two passes: the first pays the compile cost, the second confirms the graph
    # is captured and replayable on this thread.
    for _ in range(2):
        with torch.inference_mode():
            model.predict_step(batch, make_asset=False, progress_callback=None)


def load(options: Optional[dict] = None) -> dict:
    """Load the model pipeline into this process (a worker's ``load``)."""
    from src.data.transform import Transform
    from src.tokenizer.parse import get_tokenizer
    from src.server.spec import get_model

    options = options or {}
    model_ckpt = options.get("model_ckpt") or config.MODEL_CKPT
    hf_path = options.get("hf_path") or config.HF_PATH
    threads = options.get("cpu_threads")
    if threads:
        torch.set_num_threads(int(threads))

    def load_model_fn():
        model = get_model(model_ckpt, hf_path=hf_path, device=DEVICE)
        tokenizer = get_tokenizer(**model.tokenizer_config)
        transform = Transform.parse(**model.transform_config["predict_transform"])
        return model, tokenizer, transform

    state.model, state.tokenizer, state.transform = run_startup_stage(
        f"loading model from {model_ckpt}", load_model_fn
    )
    logger.info("Pipeline ready (fully resident in VRAM)")

    # Prewarm the fast-decode graph so the first real request doesn't pay the
    # one-time compile cost; on this (the serving) thread.
    if config.FAST_DECODE:
        run_startup_stage("prewarming fast decode", _prewarm_fast_decode)

    info = {"model_ckpt": model_ckpt, "device": DEVICE}
    if torch.cuda.is_available() and DEVICE.startswith("cuda"):
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20)
    return info


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
@contextmanager
def seeded_torch_rng(seed: int | None):
    """Scope a seed to torch/CUDA RNG only, restoring global state on exit.

    HuggingFace ``generate`` samples from the *global* torch RNG (it accepts no
    per-call generator), so a request's seed has to be applied globally. A
    worker process runs one request at a time, so seeding here is race-free.
    Saving/restoring the state additionally prevents a seeded request from
    perturbing the next request's sampling. numpy/python RNG is left untouched.
    """
    if seed is None:
        yield
        return
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _report_sample_progress(progress_callback, sample_count: int, sample_idx: int,
                            phase: str) -> None:
    """Map a per-sample export phase to a percentage in the 85–99 band."""
    if not progress_callback:
        return
    sample_progress_start = 85
    sample_progress_span = 14
    base = sample_progress_start + (sample_progress_span * sample_idx) // sample_count
    done = sample_progress_start + (sample_progress_span * (sample_idx + 1)) // sample_count
    mid = base + max(1, (done - base) // 2)
    sample_label = f"sample {sample_idx + 1}/{sample_count}"
    if phase == "start":
        if sample_idx > 0:
            return
        step = f"exporting {sample_label}"
        percent = base
    elif phase == "exported":
        step = f"exported {sample_label}"
        percent = min(mid, done - 1)
    elif phase == "complete":
        step = f"finished {sample_label}"
        percent = done
    else:
        raise ValueError(f"unknown sample progress phase: {phase}")
    progress_callback(percent, step)


def prepare_inputs(input_path: Path, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback=None):
    """Stage 1 (CPU): parse the input file into a single model batch."""
    from src.data.dataset import DatasetConfig, RigDatasetModule

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(5, "building dataset")

    datapath = {
        "data_name": None,
        "loader": "glb_untrusted",
        "filepaths": {"articulation": [str(input_path)]},
    }
    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=DEVICE.startswith("cuda"),
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=state.transform,
        tokenizer=state.tokenizer,
        process_fn=state.model._process_fn,
    )
    dataloader = module.predict_dataloader()["articulation"]
    cancellation.raise_if_cancelled()

    batch = next(iter(dataloader), None)
    if batch is None:
        raise RuntimeError("No data in dataloader")
    return batch


def run_inference(batch: dict, params: GenParams, request_id: str,
                  cancellation: Cancellation, progress_callback=None):
    """Stage 2 (GPU): run model sampling, returning a list of TokenRigResult."""
    from torch import Tensor
    from src.model.tokenrig import TokenRigResult

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(10, "running inference")

    batch = {
        k: v.to(state.model.device, non_blocking=True) if isinstance(v, Tensor) else v
        for k, v in batch.items()
    }
    # Inputs are plain meshes: the model always generates skeleton + skin.
    batch.pop("skeleton_tokens", None)
    batch.pop("skeleton_mask", None)

    generate_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        top_k=params.top_k,
        top_p=params.top_p,
        temperature=params.temperature,
        repetition_penalty=params.repetition_penalty,
        num_beams=params.num_beams,
        num_samples=params.num_samples,
        do_sample=True,
        use_cache=config.USE_GENERATION_CACHE,
    )

    # Fast decode requires single-beam search; beam requests fall back to the
    # original flash-attention + dynamic-cache path.
    fast_decode = config.FAST_DECODE and params.num_beams == 1
    if fast_decode:
        try:
            compile_config = _get_fast_decode_compile_config()
        except Exception:
            logger.warning("[%s] fast decode unavailable (CompileConfig import failed); "
                           "falling back to flash/dynamic path", request_id)
            fast_decode = False
    if fast_decode:
        _set_llm_attention(state.model, _SDPA_ATTN_IMPL)
        generate_kwargs.update(cache_implementation="static", compile_config=compile_config)
        logger.info("[%s] fast decode enabled (sdpa + static cache + %s)",
                    request_id, config.FAST_DECODE_BACKEND)
    elif config.FAST_DECODE:
        # A previous fast request may have switched the model to SDPA; restore
        # flash-attention for the original path.
        _set_llm_attention(state.model, _DEFAULT_ATTN_IMPL)

    batch["generate_kwargs"] = generate_kwargs

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(10, "model sampling")

    with seeded_torch_rng(params.seed), torch.inference_mode():
        preds: List[TokenRigResult] = state.model.predict_step(
            batch, make_asset=True, progress_callback=progress_callback or None,
        )["results"]

    preds = [
        dataclasses.replace(pred, cond=None, cond_latents=None,
                            input_ids=None, output_ids=None, skin_pred=None)
        for pred in preds
    ]

    del batch
    if config.EMPTY_CACHE_AFTER_INFERENCE and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    cancellation.raise_if_cancelled()
    return preds


def export_samples(preds: list, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback,
                   tmp_output_dir: Path) -> list[bytes]:
    """Stage 3 (CPU): export each predicted asset to a GLB."""
    from src.rig_package.glb_io import InputLimits
    from src.rig_package.parser.glb import transfer_rigging

    limits = InputLimits.from_env()
    sample_count = max(len(preds), 1)
    glbs: list[bytes] = []

    for sample_idx, pred in enumerate(preds):
        cancellation.raise_if_cancelled()
        _report_sample_progress(progress_callback, sample_count, sample_idx, "start")

        asset = pred.asset
        assert asset is not None
        collapsed_joints = asset.collapse_near_parent_joints()
        if collapsed_joints:
            logger.info("[%s] sample=%d collapsed near-parent skeleton joints: %s",
                        request_id, sample_idx, ", ".join(collapsed_joints))

        sample_out_path = tmp_output_dir / f"sample_{sample_idx}.glb"
        sample_out_path.parent.mkdir(parents=True, exist_ok=True)
        transfer_rigging(
            source_asset=asset,
            target_path=asset.path,
            export_path=str(sample_out_path),
            group_per_vertex=4,
            # Shift the whole scene so its bottom sits at Z=0.
            auto_ground=True,
            limits=limits,
        )
        glbs.append(sample_out_path.read_bytes())
        cancellation.raise_if_cancelled()
        _report_sample_progress(progress_callback, sample_count, sample_idx, "exported")

    return glbs


class ProgressReporter:
    """Per-request progress/timing; checks for cancellation at every report."""

    LOG_SKIPPED_PREFIXES = ("model sampling", "decoding joints")

    def __init__(self, request_id: str, cancellation: Cancellation, send=None):
        self.request_id = request_id
        self.cancellation = cancellation
        self.send = send
        self.started_at = time.monotonic()
        self.last_logged_at = self.started_at

    def report(self, percent: int, stage: str) -> None:
        self.cancellation.raise_if_cancelled()
        now = time.monotonic()
        elapsed = round(now - self.started_at, 2)
        if not stage.startswith(self.LOG_SKIPPED_PREFIXES):
            logger.info("[%s] progress=%d%% stage=%s elapsed=%.2fs delta=%.2fs",
                        self.request_id, percent, stage, elapsed, now - self.last_logged_at)
            self.last_logged_at = now
        if self.send is not None:
            self.send({"progress": percent, "step": stage, "elapsed_sec": elapsed})
        self.cancellation.raise_if_cancelled()


# Stage functions run_job calls; tests replace them.
STAGES = {"prepare": prepare_inputs, "infer": run_inference, "export": export_samples}


def run_job(params: dict, upload: bytes, cancellation: Cancellation, progress=None) -> list[bytes]:
    """One request (a worker's ``run``): the rigged GLB bytes, one per sample.

    *params* holds the :class:`GenParams` fields plus ``suffix`` (the upload's
    file extension) and ``request_id``.
    """
    params = dict(params)
    suffix = params.pop("suffix")
    request_id = params.pop("request_id", "-")
    gen = GenParams(**params)
    reporter = ProgressReporter(request_id, cancellation, progress)

    tmp_input_dir = Path(tempfile.mkdtemp(prefix="skintokens_input_"))
    tmp_output_dir = Path(tempfile.mkdtemp(prefix="skintokens_output_"))
    # The client's filename is never used as a path: only its suffix is.
    input_path = tmp_input_dir / f"input{suffix}"
    try:
        input_path.write_bytes(upload)
        del upload
        batch = STAGES["prepare"](input_path, gen, request_id, cancellation, reporter.report)
        preds = STAGES["infer"](batch, gen, request_id, cancellation, reporter.report)
        del batch
        glbs = STAGES["export"](preds, gen, request_id, cancellation, reporter.report,
                                tmp_output_dir)
        del preds
        reporter.report(100, "complete")
        return glbs
    finally:
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
        shutil.rmtree(tmp_output_dir, ignore_errors=True)


SERVICE = ServiceSpec("skintokens", load, run_job)
