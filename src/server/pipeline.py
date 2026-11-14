"""The model pipeline as a worker process runs it.

Each worker process (:mod:`src.server.worker`) loads its own copy of the model
with :func:`load` and serves one request at a time with :func:`run_job`:
parse the upload, sample the rig on the GPU, export the GLB(s). Nothing here is
shared between requests running in different workers, so the stages need no
locks: the parallelism comes from running several workers.

.. note::

    Decoding is fixed to greedy single-beam search, so a request's only
    randomness is where points are sampled on the mesh. Every stage runs under
    seeds derived from the request's ``seed`` (:mod:`src.data.seeding`), so the
    same input, ``seed`` and ``num_samples`` give the same GLBs on the same GPU
    and software stack. Different GPUs or library versions may select different
    kernels and so may round differently.
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
import threading
import time
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

from src.model.attention import LLM_ATTN_IMPL as _DEFAULT_ATTN_IMPL
from src.model import decode_attention

# The fast decode path's attention: SDPA for prefill, a length-aware kernel for
# the single-token decode steps over the static cache (src/model/decode_attention.py).
# Registered with transformers in load(), before any generation.
_FAST_ATTN_IMPL = decode_attention.DECODE_ATTN_IMPL

__all__ = ["GenerationCancelled", "load", "run_job", "SERVICE"]


class _State:
    model = None
    tokenizer = None
    transform = None
    release_pending = False


state = _State()

# Built lazily so a transformers build without CompileConfig disables fast decode
# gracefully rather than failing at import time.
_fast_decode_compile_config = None


def _set_llm_attention(model, impl: str) -> None:
    """Switch only the LLM transformer's attention backend.

    The mesh encoder and VAE call ``flash_attn_func`` directly (not via config),
    so they are unaffected; only Qwen3's config-dispatched attention changes.
    Flash-attention mis-masks the pre-allocated static cache (it attends over the
    unfilled slots), so the fast path runs ``_FAST_ATTN_IMPL``. No-op if already set.
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

    _set_llm_attention(model, _FAST_ATTN_IMPL)
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
    decode_attention.register()
    if config.DETERMINISTIC:
        from src.data.seeding import enable_deterministic_kernels
        enable_deterministic_kernels()
    logger.info("deterministic kernels: %s", config.DETERMINISTIC)
    # generate() checks for its end one step late instead of syncing every token.
    from src.model.grammar import enable_cuda_deferred_stop
    logger.info("deferred stop check on CUDA: %s", enable_cuda_deferred_stop())

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


@dataclasses.dataclass(frozen=True)
class Upload:
    """An uploaded file, kept in memory (never written to disk). ``name`` is
    chosen by the server (``input`` + the validated suffix); the client's
    filename is used for nothing."""
    data: bytes
    name: str


def _load_input(source, limits):
    """Parse ``source`` (an :class:`Upload` or a file path) as a scene."""
    from src.rig_package.glb_io import load_scene
    if isinstance(source, Upload):
        return load_scene(source.data, limits=limits, name=source.name)
    return load_scene(source, limits=limits)


def prepare_inputs(source, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback=None):
    """Stage 1 (CPU): parse the input (an :class:`Upload`, or a file path) into
    a single model batch holding ``params.num_samples`` point clouds, cloud
    ``i`` sampled under ``derive_seed(params.seed, i)``.

    The batch carries the parsed ``scene`` too, so export need not parse the
    file again; :func:`run_job` takes it out before inference.
    """
    from src.data.seeding import sample_point_clouds, seeded_rng
    from src.model.spec import ModelInput
    from src.rig_package.glb_io import InputLimits
    from src.rig_package.parser.glb import GlbParser

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(5, "building dataset")

    transform = state.transform
    scene = _load_input(source, InputLimits.from_env())
    # The model only sees normals at its sampled points (the export copies
    # the file's own). When the transform moves the vertices, normals are
    # not loaded and the sampler computes them at its points only.
    asset = GlbParser.asset_from_scene(scene, include_normals=not transform.rebuilds_normals)
    # The class the model is conditioned on (the dataset key the dataloader
    # path used to derive it from).
    asset.cls = "articulation"
    asset.path = source.name if isinstance(source, Upload) else str(source)
    cancellation.raise_if_cancelled()

    with seeded_rng(params.seed):
        transform.apply(asset=asset, sample=False)
    cancellation.raise_if_cancelled()

    # One cloud per sample (the transform's own single cloud is not drawn).
    vertices, normals = sample_point_clouds(asset, transform.sampler,
                                            params.seed, params.num_samples)
    return {
        "vertices": vertices.unsqueeze(0),
        "normals": normals.unsqueeze(0),
        "cls": [asset.cls],
        "model_input": [ModelInput(asset=asset)],
        "scene": scene,
    }


def run_inference(batch: dict, params: GenParams, request_id: str,
                  cancellation: Cancellation, progress_callback=None):
    """Stage 2 (GPU): decode one rig per point cloud, returning a list of TokenRigResult."""
    from torch import Tensor
    from src.data.seeding import seeded_rng
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

    # Decoding is greedy and single-beam (fixed in TokenRig.generate).
    generate_kwargs = dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        use_cache=config.USE_GENERATION_CACHE,
    )

    fast_decode = config.FAST_DECODE
    if fast_decode:
        try:
            compile_config = _get_fast_decode_compile_config()
        except Exception:
            logger.warning("[%s] fast decode unavailable (CompileConfig import failed); "
                           "falling back to flash/dynamic path", request_id)
            fast_decode = False
    if fast_decode:
        _set_llm_attention(state.model, _FAST_ATTN_IMPL)
        generate_kwargs.update(cache_implementation="static", compile_config=compile_config)
        logger.info("[%s] fast decode enabled (%s attention + static cache + %s)",
                    request_id, decode_attention.DECODE_ATTN_MODE, config.FAST_DECODE_BACKEND)
    elif config.FAST_DECODE:
        # Fast decode was requested but is unavailable; the prewarm may have
        # switched the model's attention, so restore the default one.
        _set_llm_attention(state.model, _DEFAULT_ATTN_IMPL)

    batch["generate_kwargs"] = generate_kwargs

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(10, "model sampling")

    with seeded_rng(params.seed), torch.inference_mode():
        # The skin stays at the sampled points: export interpolates only the
        # few influences per vertex it writes (see export_samples).
        preds: List[TokenRigResult] = state.model.predict_step(
            batch, make_asset=True, progress_callback=progress_callback or None,
            interpolate_skin=False,
        )["results"]

    preds = [
        dataclasses.replace(pred, cond=None, cond_latents=None,
                            input_ids=None, output_ids=None, skin_pred=None)
        for pred in preds
    ]

    del batch
    # The memory is released after the reply is sent (:func:`release_memory`),
    # not here, where it would delay the export and the response.
    state.release_pending = True

    cancellation.raise_if_cancelled()
    return preds


def release_memory() -> None:
    """Hand cached memory back after a request (a worker's ``idle`` hook).

    Runs once the reply has been sent, so it costs the next request nothing
    unless that one is already waiting.
    """
    if not state.release_pending:
        return
    state.release_pending = False
    if config.EMPTY_CACHE_AFTER_INFERENCE and torch.cuda.is_available():
        started = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("memory released in %.0fms", (time.perf_counter() - started) * 1000)


GROUP_PER_VERTEX = 4  # influences per vertex in the exported GLB


def _skin_device() -> Optional[str]:
    """Where to interpolate the skin: the model's CUDA device, else CPU threads."""
    return DEVICE if DEVICE.startswith("cuda") and torch.cuda.is_available() else None


def export_samples(preds: list, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback,
                   tmp_output_dir: Optional[Path] = None, scene=None,
                   source=None) -> list[bytes]:
    """Stage 3 (CPU): export each predicted asset to a GLB (sample ``i`` under
    ``derive_seed(params.seed, i)``: the rig transfer subsamples points).

    ``scene`` is the input already parsed by :func:`prepare_inputs`; without it
    ``source`` (an :class:`Upload` or a path, else the asset's own path) is
    parsed once here. ``tmp_output_dir`` is unused (GLBs are built in memory)
    and kept for callers.
    """
    from src.data.seeding import derive_seed, seeded_rng
    from src.rig_package.glb_io import InputLimits, SparseSkin
    from src.rig_package.parser.glb import transfer_rigging
    from src.rig_package.skin_interp import interpolate_top_k

    sample_count = max(len(preds), 1)
    glbs: list[bytes] = []
    if scene is None and preds:
        scene = _load_input(source or preds[0].asset.path, InputLimits.from_env())

    for sample_idx, pred in enumerate(preds):
        cancellation.raise_if_cancelled()
        _report_sample_progress(progress_callback, sample_count, sample_idx, "start")

        asset = pred.asset
        assert asset is not None
        # Merging a collapsed joint's skin into its ancestor commutes with the
        # (linear) interpolation onto the mesh, so with the skin still at the
        # sampled points, it is merged there, on far fewer rows.
        collapsed_joints = asset.collapse_near_parent_joints()
        if collapsed_joints:
            logger.info("[%s] sample=%d collapsed near-parent skeleton joints: %s",
                        request_id, sample_idx, ", ".join(collapsed_joints))

        skin = None
        sampled_skin = (asset.sampled_vertex_groups or {}).get("skin")
        if asset.skin is None and sampled_skin is not None:
            ids, weights = interpolate_top_k(
                asset.vertices, asset.sampled_vertices, sampled_skin,
                top_k=GROUP_PER_VERTEX, device=_skin_device())
            skin = SparseSkin(ids, weights)

        with seeded_rng(derive_seed(params.seed, sample_idx)):
            glb = transfer_rigging(
                source_asset=asset,
                scene=scene,
                skin=skin,
                export_path=None,  # build the GLB in memory
                group_per_vertex=GROUP_PER_VERTEX,
                # Shift the whole scene so its bottom sits at Z=0.
                auto_ground=True,
            )
        glbs.append(glb)
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
    gen = GenParams(**params).with_seed()
    logger.info("[%s] seed=%d num_samples=%d", request_id, gen.seed, gen.num_samples)
    reporter = ProgressReporter(request_id, cancellation, progress)

    # Parsed straight from memory. The client's filename is used for nothing:
    # only its (validated) suffix, which tells GLB from glTF.
    source = Upload(upload, f"input{suffix}")
    del upload
    batch = STAGES["prepare"](source, gen, request_id, cancellation, reporter.report)
    scene = batch.pop("scene", None) if isinstance(batch, dict) else None
    preds = STAGES["infer"](batch, gen, request_id, cancellation, reporter.report)
    del batch
    glbs = STAGES["export"](preds, gen, request_id, cancellation, reporter.report,
                            scene=scene, source=source)
    del preds, scene, source
    reporter.report(100, "complete")
    return glbs


SERVICE = ServiceSpec("skintokens", load, run_job, release_memory)
