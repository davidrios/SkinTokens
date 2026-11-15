"""The GPU stage of a request, as a GPU worker process runs it.

Each GPU worker process (:mod:`src.server.worker`) loads its own copy of the
model with :func:`load` and serves one request's ``infer`` stage at a time
with :func:`run_job`: sample the rig from the point clouds ``prepare`` drew,
and interpolate its skin onto the mesh vertices. Parsing the upload and
exporting the result run in CPU worker processes (:mod:`src.server.stages`),
so they overlap other requests' inference. Nothing here is shared between
requests running in different workers, so the stage needs no locks.

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
import sys

# expandable_segments (less fragmentation, so a lower reserved peak) needs the
# CUDA VMM APIs, so it is enabled on native Linux only: Windows (WDDM) ignores
# it, and under WSL2 (/dev/dxg, e.g. Docker Desktop) recent torch fails every
# allocation with "CUDA driver error: unknown error".
if sys.platform.startswith("linux") and not os.path.exists("/dev/dxg"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import dataclasses
import gc
import logging
import threading
import time
from typing import List, Optional

import torch

from src.server import config
from src.server.config import GenParams
from src.server.stages import (InferInput, Prediction, ProgressReporter, finish_skin, pack,
                               unpack)
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
            # Triton-free CUDA-graph capture (no fusion), with the weights and KV
            # cache as static inputs rather than copied into the graph each step
            # (src/model/cudagraph_backend.py). The backend rejects the ``mode``
            # argument, so it must be left unset.
            from src.model import cudagraph_backend
            _fast_decode_compile_config = CompileConfig(
                backend=cudagraph_backend.register(), mode=None, fullgraph=False
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

    # The CPU workers build the same input transform from this (they load no model).
    info = {"model_ckpt": model_ckpt, "device": DEVICE,
            "predict_transform": state.model.transform_config["predict_transform"]}
    if torch.cuda.is_available() and DEVICE.startswith("cuda"):
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20)
    return info


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
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
        logger.debug("[%s] fast decode enabled (%s attention + static cache + %s)",
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
    # not here, where it would delay the reply.
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
        gc.collect()
        torch.cuda.empty_cache()


def _skin_device() -> Optional[str]:
    """Where to interpolate the skin: the model's CUDA device, else CPU threads."""
    return DEVICE if DEVICE.startswith("cuda") and torch.cuda.is_available() else None


def run_job(params: dict, payload: bytes, cancellation: Cancellation, progress=None) -> bytes:
    """One request's GPU stage (a GPU worker's ``run``): sample one rig per
    point cloud and interpolate each skin onto the mesh vertices on the GPU.

    *params* holds the :class:`GenParams` fields (with the request's seed)
    plus ``request_id``; *payload* is the packed :class:`InferInput` from the
    ``prepare`` stage. Returns the packed list of :class:`Prediction`, for the
    ``export`` stage (:mod:`src.server.stages`).
    """
    params = dict(params)
    request_id = params.pop("request_id", "-")
    gen = GenParams(**params)
    if gen.seed is None:
        raise ValueError("the gateway sends every stage a seed")
    reporter = ProgressReporter(request_id, cancellation, progress)

    from src.model.spec import ModelInput

    infer_input: InferInput = unpack(payload)
    del payload
    batch = {
        "vertices": torch.from_numpy(infer_input.vertices).unsqueeze(0),
        "normals": torch.from_numpy(infer_input.normals).unsqueeze(0),
        "cls": [infer_input.cls],
        "model_input": [ModelInput(asset=infer_input.mesh)],
    }
    del infer_input
    preds = STAGES["infer"](batch, gen, request_id, cancellation, reporter.report)
    del batch

    # The skin is interpolated here, where the GPU is, so the CPU export gets
    # only each vertex's few heaviest influences (and the sampled skin stays).
    device = _skin_device()
    predictions = []
    for sample_idx, pred in enumerate(preds):
        cancellation.raise_if_cancelled()
        asset = pred.asset
        skin = finish_skin(asset, request_id, sample_idx, device=device)
        if skin is not None:
            asset.sampled_vertices = None
            asset.sampled_vertex_groups = None
        predictions.append(Prediction(asset=asset, skin=skin))
    del preds
    reporter.report(85, "inference done")
    return pack(predictions)


# Stage functions run_job calls; tests replace them.
STAGES = {"infer": run_inference}

SERVICE = ServiceSpec("skintokens", load, run_job, release_memory)
