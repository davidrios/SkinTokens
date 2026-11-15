"""The GPU stage of a request, as a GPU worker process runs it.

Each GPU worker process (:mod:`src.server.worker`) loads its own copy of the
model with :func:`load` and runs requests' ``infer`` stage: sample the rig
from the point clouds ``prepare`` drew, and interpolate its skin onto the mesh
vertices. Parsing the upload and exporting the result run in CPU worker
processes (:mod:`src.server.stages`), so they overlap other requests'
inference. Nothing here is shared between requests running in different
workers, so the stage needs no locks.

With fast decode on CUDA, a worker batches its requests continuously
(:func:`serve`, :mod:`src.model.batch_decode`): it accepts up to
``SKINTOKENS_BATCH_SLOTS`` (1, 2, 3 or 4) requests at once and decodes up to that
many point clouds together, each joining when a slot frees and leaving when
it ends. Otherwise it serves one request at a time (:func:`run_job`).

.. note::

    Decoding is fixed to greedy single-beam search, so a request's only
    randomness is where points are sampled on the mesh, under seeds derived
    from the request's ``seed`` (:mod:`src.data.seeding`). Results are
    reproducible only as far as the GPU kernels round the same way: different
    GPUs or library versions may select different kernels, and with
    continuous batching the kernels a row runs also depend on how many other
    rows were decoding beside it. Greedy decoding then breaks near-ties (e.g.
    between two adjacent coordinate bins) differently, so the same input and
    ``seed`` may give slightly different rigs from run to run.
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

import collections
import dataclasses
import gc
import logging
import threading
import time
from typing import Any, List, Optional

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
    decoder = None  # the continuous-batching decoder, when serve() runs
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
            use_cache=True,
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

    # Captured on this (the serving) thread, so the first real request doesn't
    # pay the one-time compile cost.
    slots = int(options.get("batch_slots", config.BATCH_SLOTS))
    state.decoder = run_startup_stage("capturing the batched decoder",
                                      lambda: _build_decoder(slots))
    if state.decoder is None and config.FAST_DECODE:
        run_startup_stage("prewarming fast decode", _prewarm_fast_decode)

    # The CPU workers build the same input transform from this (they load no model).
    info = {"model_ckpt": model_ckpt, "device": DEVICE,
            "predict_transform": state.model.transform_config["predict_transform"]}
    if state.decoder is not None:
        # The gateway sends this many requests at once (worker.py).
        info["slots"] = state.decoder.slots
        info["kv_cache_mb"] = round(state.decoder.kv_cache_bytes() / 2**20)
    if torch.cuda.is_available() and DEVICE.startswith("cuda"):
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20)
    return info


def _build_decoder(slots: int):
    """The continuous-batching decoder with its graphs captured, or None when
    this worker serves one request at a time (no CUDA, fast decode off, or a
    tokenizer without the GPU grammar)."""
    from src.model import grammar
    from src.model.batch_decode import BatchDecoder

    model = state.model
    if slots < 1:
        logger.info("continuous batching off (SKINTOKENS_BATCH_SLOTS=0)")
        return None
    if not (config.FAST_DECODE and DEVICE.startswith("cuda") and torch.cuda.is_available()):
        logger.info("continuous batching off: needs fast decode on CUDA")
        return None
    if not (grammar.gpu_grammar_enabled() and grammar.supports(model.tokenizer)):
        logger.info("continuous batching off: needs the GPU grammar")
        return None
    _set_llm_attention(model, _FAST_ATTN_IMPL)
    # The prompt's length does not depend on the geometry (the mesh encoder
    # emits a fixed number of condition tokens), so a random cloud sizes it.
    n_pts = 8192
    vertices = torch.rand(1, n_pts, 3, device=model.device) * 2.0 - 1.0
    normals = torch.nn.functional.normalize(torch.randn(1, n_pts, 3, device=model.device), dim=-1)
    prompt = model.prepare_prompts(vertices, normals, cls="prewarm")[0]
    # As on the one-request path: "inductor" fuses the step's pointwise ops
    # (Triton needed), "cudagraphs" captures them as they are.
    compile_step = config.FAST_DECODE_BACKEND == "inductor" and decode_attention._has_triton()
    decoder = BatchDecoder(model, slots, config.MAX_NEW_TOKENS,
                           prompt_length=prompt["inputs_embeds"].shape[0],
                           compile_step=compile_step)
    decoder.capture()
    # One row through prefill and a few steps, so neither compiles on a request.
    decoder.insert("prewarm", prompt["inputs_embeds"], prompt["start_tokens"])
    for _ in range(16):
        decoder.step()
    decoder.remove("prewarm")
    torch.cuda.synchronize()
    logger.info("continuous batching: %d slots, buckets %s, KV cache %.0f MiB, step %s",
                decoder.slots, decoder.buckets, decoder.kv_cache_bytes() / 2**20,
                "compiled (inductor)" if compile_step else "eager")
    return decoder


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
        use_cache=True,
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


# --------------------------------------------------------------------------- #
# Continuous batching
# --------------------------------------------------------------------------- #
# Decode steps between progress reports (and their cancellation checks).
PROGRESS_EVERY_STEPS = 16


@dataclasses.dataclass
class _Request:
    """A request in :func:`serve`: its point clouds decode as separate rows."""
    job: Any
    request_id: str
    seed: int
    reporter: ProgressReporter
    mesh: Any
    vertices: Any  # (S, P, 3) numpy, the sampled points
    prompts: List[Optional[dict]]  # per sample (TokenRig.prepare_prompts)
    predictions: List[Optional[Prediction]]
    percent: int = 10

    @property
    def done(self) -> bool:
        return all(p is not None for p in self.predictions)


class _Scheduler:
    """Feeds the requests' rows to the decoder and finishes them as they end."""

    def __init__(self, source) -> None:
        self.source = source
        self.decoder = state.decoder
        self.requests: dict[int, _Request] = {}
        # Rows waiting for a slot, as (job id, sample index), FIFO.
        self.waiting: "collections.deque[tuple[int, int]]" = collections.deque()
        self.steps = 0

    def run(self) -> None:
        while True:
            busy = bool(self.decoder.active or self.waiting)
            if not busy:
                release_memory()
            job = self.source.next(block=not busy)
            while job is not None:
                self.admit(job)
                job = self.source.next(block=False)
            for request in list(self.requests.values()):
                if request.job.cancellation.cancelled:
                    self.drop(request, GenerationCancelled(request.job.cancellation.reason))
            self.fill_slots()
            if self.decoder.active:
                for finished in self.decoder.step():
                    self.finish_row(finished)
                self.steps += 1
                if self.steps % PROGRESS_EVERY_STEPS == 0:
                    self.report_progress()

    def admit(self, job) -> None:
        """Encode a new request's point clouds and queue its rows."""
        from src.data.seeding import seeded_rng

        params = dict(job.params)
        request_id = params.pop("request_id", "-")
        reporter = ProgressReporter(request_id, job.cancellation, job.progress)
        try:
            gen = GenParams(**params)
            if gen.seed is None:
                raise ValueError("the gateway sends every stage a seed")
            infer_input: InferInput = unpack(job.upload)
            job.upload = None
            reporter.report(10, "running inference")
            device = state.model.device
            vertices = torch.from_numpy(infer_input.vertices).to(device)
            normals = torch.from_numpy(infer_input.normals).to(device)
            with seeded_rng(gen.seed):
                prompts = state.model.prepare_prompts(vertices, normals, infer_input.cls)
            del vertices, normals
            request = _Request(job, request_id, gen.seed, reporter, infer_input.mesh,
                               infer_input.vertices, prompts, [None] * len(prompts))
            reporter.report(10, "model sampling")
        except Exception as exc:
            job.fail(exc)
            return
        self.requests[job.id] = request
        self.waiting.extend((job.id, i) for i in range(len(prompts)))
        state.release_pending = True

    def fill_slots(self) -> None:
        while self.decoder.free and self.waiting:
            job_id, sample = self.waiting.popleft()
            request = self.requests.get(job_id)
            if request is None:
                continue  # dropped meanwhile
            prompt = request.prompts[sample]
            try:
                self.decoder.insert((job_id, sample), prompt["inputs_embeds"],
                                    prompt["start_tokens"])
            except ValueError as exc:
                self.drop(request, exc)

    def finish_row(self, finished) -> None:
        """Decode an ended row's skin, attach the rig, and reply once the
        request's every row is done."""
        from src.data.seeding import seeded_rng

        job_id, sample = finished.key
        request = self.requests.get(job_id)
        if request is None:
            return
        prompt, request.prompts[sample] = request.prompts[sample], None
        # Progress past sampling only once the request's last row ends (its
        # other rows' sampling still reports below 74 %).
        last = sum(p is None for p in request.predictions) == 1
        if last:
            report = request.reporter.report
        else:
            def report(percent, stage):
                request.job.cancellation.raise_if_cancelled()
        try:
            report(max(request.percent, 74), "preparing for export...")
            model = state.model
            with seeded_rng(request.seed):
                result = model.result_from_ids(
                    prompt["start_tokens"], finished.generated_ids.to(model.device),
                    prompt["cond"], prompt["cond_latents"], progress_callback=report)
            del prompt
            model.attach_asset(result, request.mesh, request.vertices[sample].astype("float32"),
                               interpolate_skin=False)
            asset = result.asset
            skin = finish_skin(asset, request.request_id, sample, device=_skin_device())
            if skin is not None:
                asset.sampled_vertices = None
                asset.sampled_vertex_groups = None
            request.predictions[sample] = Prediction(asset=asset, skin=skin)
            if request.done:
                request.reporter.report(85, "inference done")
                del self.requests[job_id]
                request.job.reply(pack(request.predictions))
        except Exception as exc:
            self.drop(request, exc)

    def drop(self, request: _Request, exc: BaseException) -> None:
        """Fail *request*: free its rows' slots and answer with *exc*."""
        self.requests.pop(request.job.id, None)
        for sample in range(len(request.prompts)):
            self.decoder.remove((request.job.id, sample))
        request.prompts = []
        request.job.fail(exc)

    def report_progress(self) -> None:
        """Each request's sampling progress (10-73 %), from the rows' status."""
        tokens_per_skin = state.model.tokens_per_skin
        default_total = max(240, min(config.MAX_NEW_TOKENS, 512))
        for request in list(self.requests.values()):
            ratios, tokens = [], 0
            for sample, prediction in enumerate(request.predictions):
                if prediction is not None:
                    ratios.append(1.0)
                    continue
                status = self.decoder.progress((request.job.id, sample))
                if status is None:
                    ratios.append(0.0)
                    continue
                generated = status["generated"]
                total = (status["switch_len"] + status["bones"] * tokens_per_skin + 1
                         if status["in_skin"] else default_total)
                ratios.append(min(1.0, generated / max(total, generated, 1)))
                tokens += generated
            percent = min(73, 10 + int(63 * sum(ratios) / max(1, len(ratios))))
            if percent > request.percent:
                request.percent = percent
                try:
                    request.reporter.report(percent, f"model sampling ({tokens} tokens)")
                except GenerationCancelled as exc:
                    self.drop(request, exc)


def serve(source) -> None:
    """A batching GPU worker's loop (its spec's ``serve``): runs the jobs the
    gateway sends, up to the decoder's slots at once, until it is stopped."""
    _Scheduler(source).run()


# Stage functions run_job calls; tests replace them.
STAGES = {"infer": run_inference}

SERVICE = ServiceSpec("skintokens", load, run_job, release_memory, serve)
