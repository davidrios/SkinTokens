"""Server configuration from ``SKINTOKENS_*`` environment variables.

Free of torch/model imports: the gateway process loads no model, and worker
processes read the same values when they import :mod:`src.server.pipeline`.
"""
from __future__ import annotations

import logging
import os
import secrets

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

# Worker processes, each with its own copy of the model on the GPU; up to this
# many requests run fully in parallel (parse, inference and export).
WORKERS = env_int("SKINTOKENS_WORKERS", 2, 1, 32)
# CPU threads per worker for torch/OpenMP (0 = an equal share of the cores).
WORKER_CPU_THREADS = env_int("SKINTOKENS_WORKER_CPU_THREADS", 0, 0, 256)

MAX_NUM_SAMPLES = env_int("SKINTOKENS_MAX_NUM_SAMPLES", 1, 1, 8)
MAX_NEW_TOKENS = env_int("SKINTOKENS_MAX_NEW_TOKENS", 2048, 128, 4096)
# Uploads are untrusted: capped in size here and parsed with resource limits
# (SKINTOKENS_INPUT_* env vars, see src/rig_package/glb_io/limits.py), never
# allowed to reference files outside the upload itself.
MAX_UPLOAD_MB = upload_limit_mb()
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
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


def worker_cpu_threads(workers: int) -> int:
    """Torch/OpenMP threads per worker: configured, or an equal share of the cores."""
    if WORKER_CPU_THREADS:
        return WORKER_CPU_THREADS
    return max(1, (os.cpu_count() or 1) // max(1, workers))


class GenParams(BaseModel):
    """What a request may choose. Decoding itself is fixed (greedy, one beam),
    so the outputs are a function of the input, ``seed`` and ``num_samples``:
    sample ``i`` is decoded from a point cloud drawn under a seed derived from
    ``(seed, i)`` (see :mod:`src.data.seeding`)."""
    num_samples: int = Field(default=1, ge=1, le=MAX_NUM_SAMPLES)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)

    def with_seed(self) -> "GenParams":
        """These parameters with a seed: the requested one, else a random one."""
        if self.seed is not None:
            return self
        return self.model_copy(update={"seed": secrets.randbits(63)})
