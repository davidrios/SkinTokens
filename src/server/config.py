"""Server configuration from ``SKINTOKENS_*`` environment variables.

Free of torch/model imports: the gateway process loads no model, and worker
processes read the same values when they import :mod:`src.server.pipeline`.
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Literal

from pydantic import BaseModel, Field

from src.data.seeding import MAX_SEED
from src.rig_package.glb_io.limits import upload_limit_mb

logger = logging.getLogger("skintokens.serve")


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value < minimum or value > maximum:
        clamped = min(max(value, minimum), maximum)
        logger.warning("Clamped %s=%d to %d (allowed %d..%d)",
                       name, value, clamped, minimum, maximum)
        return clamped
    return value


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODEL_CKPT = os.environ.get(
    "SKINTOKENS_MODEL_CKPT",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
)
HF_PATH = os.environ.get("SKINTOKENS_HF_PATH") or None
STARTUP_HEARTBEAT_SEC = max(1.0, float(os.environ.get("SKINTOKENS_STARTUP_HEARTBEAT_SEC", "15")))
SUPPORTED_EXT = {".glb", ".gltf"}

# GPU worker processes per visible GPU, each with its own copy of the model;
# the pool starts this many on every GPU (pinned round-robin), so up to
# WORKERS_PER_GPU * (visible GPUs) requests are sampled on the GPUs at once.
WORKERS_PER_GPU = env_int("SKINTOKENS_WORKERS", 2, 1, 32)
# CPU worker processes (no model, no GPU) that parse uploads and export
# results, beside the GPU workers' inference (0 = twice the GPU workers, so
# one request's parse and another's export can both overlap each inference).
CPU_WORKERS = env_int("SKINTOKENS_CPU_WORKERS", 0, 0, 256)
# CPU threads per worker process for torch/OpenMP (0 = an equal share of the
# cores among all GPU and CPU workers).
WORKER_CPU_THREADS = env_int("SKINTOKENS_WORKER_CPU_THREADS", 0, 0, 256)

MAX_NUM_SAMPLES = env_int("SKINTOKENS_MAX_NUM_SAMPLES", 1, 1, 8)
# Result formats a request may ask for (GenParams.output). "rig" sends only the
# skeleton and skin, which the client adds to the file it uploaded
# (src/rig_package/glb_io/rig_payload.py); the whole GLB is several times
# larger and served only when SKINTOKENS_ENABLE_GLB_OUTPUT=1.
OUTPUT_FORMATS = ("rig", "glb") if env_flag("SKINTOKENS_ENABLE_GLB_OUTPUT", False) else ("rig",)
MAX_NEW_TOKENS = env_int("SKINTOKENS_MAX_NEW_TOKENS", 2048, 128, 4096)
# Uploads are untrusted: capped in size here and parsed with resource limits
# (SKINTOKENS_INPUT_* env vars, see src/rig_package/glb_io/limits.py), never
# allowed to reference files outside the upload itself.
MAX_UPLOAD_MB = upload_limit_mb()
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# WebSocket protocol bounds (the gateway is the only process clients reach).
# Text messages (the request header, cancel messages) are small JSON objects.
MAX_TEXT_MESSAGE_BYTES = 64 * 1024
MAX_FILENAME_CHARS = 255
# Seconds a client may take to send its header / its whole upload.
HEADER_TIMEOUT_SEC = env_int("SKINTOKENS_HEADER_TIMEOUT_SEC", 30, 1, 3600)
UPLOAD_TIMEOUT_SEC = env_int("SKINTOKENS_UPLOAD_TIMEOUT_SEC", 300, 1, 3600)
# Requests allowed to wait for a worker beyond the ones running. Each holds its
# upload (up to MAX_UPLOAD_MB) in the gateway, so this bounds gateway memory
# to about (total workers + MAX_QUEUE) * MAX_UPLOAD_MB; further requests are
# refused.
MAX_QUEUE = env_int("SKINTOKENS_MAX_QUEUE", 16, 0, 10000)
USE_GENERATION_CACHE = env_flag("SKINTOKENS_USE_CACHE", True)
EMPTY_CACHE_AFTER_INFERENCE = env_flag("SKINTOKENS_EMPTY_CACHE_AFTER_INFERENCE", True)
# Fast decode: SDPA attention + static KV cache + CUDA-graph capture gives ~2x
# on the autoregressive decoding stage, which dominates wall time.
FAST_DECODE = env_flag("SKINTOKENS_FAST_DECODE", True)
# Deterministic kernels, so the same (input, seed, num_samples) gives the same
# rig whichever worker serves it and across restarts; costs decode speed
# (src.data.seeding.enable_deterministic_kernels). Off: reproducible only
# within one worker process.
DETERMINISTIC = env_flag("SKINTOKENS_DETERMINISTIC", False)
# Compile backend for the fast decode path:
#   "cudagraphs" (default) — Triton-free, CUDA-graph capture only (no op fusion),
#                            works everywhere including Windows dev.
#   "inductor"             — adds pointwise fusion (RMSNorm/RoPE/casts) on top of
#                            CUDA graphs; faster, but requires Triton (available in
#                            the Linux container, not on Windows).
FAST_DECODE_BACKEND = os.environ.get("SKINTOKENS_FAST_DECODE_BACKEND", "cudagraphs").strip().lower()


def cpu_workers(gpu_workers: int) -> int:
    """The number of CPU workers: configured, or twice the GPU workers."""
    return CPU_WORKERS or 2 * max(1, gpu_workers)


def worker_cpu_threads(workers: int) -> int:
    """Torch/OpenMP threads per worker process: configured, or an equal share
    of the cores among *workers* processes."""
    if WORKER_CPU_THREADS:
        return WORKER_CPU_THREADS
    return max(1, (os.cpu_count() or 1) // max(1, workers))


class GenParams(BaseModel):
    """What a request may choose. Decoding itself is fixed (greedy, one beam),
    so the outputs are a function of the input, ``seed`` and ``num_samples``:
    sample ``i`` is decoded from a point cloud drawn under a seed derived from
    ``(seed, i)`` (see :mod:`src.data.seeding`).

    ``output`` is the result format: ``"rig"`` (a rig payload, the default) or
    ``"glb"``. The server refuses formats not in :data:`OUTPUT_FORMATS`."""
    num_samples: int = Field(default=1, ge=1, le=MAX_NUM_SAMPLES)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    output: Literal["rig", "glb"] = "rig"

    def with_seed(self) -> "GenParams":
        """These parameters with a seed: the requested one, else a random one."""
        if self.seed is not None:
            return self
        return self.model_copy(update={"seed": secrets.randbits(63)})
