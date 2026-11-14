"""Server configuration from ``SKINTOKENS_*`` environment variables.

Free of torch/model imports: the gateway process loads no model, and worker
processes read the same values when they import :mod:`src.server.pipeline`.
"""
from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

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
SUPPORTED_EXT = {".obj", ".glb", ".gltf"}

# Worker processes, each with its own copy of the model on the GPU; up to this
# many requests run fully in parallel (parse, inference and export).
WORKERS = env_int("SKINTOKENS_WORKERS", 2, 1, 32)
# CPU threads per worker for torch/OpenMP (0 = an equal share of the cores).
WORKER_CPU_THREADS = env_int("SKINTOKENS_WORKER_CPU_THREADS", 0, 0, 256)

DEFAULT_NUM_BEAMS = env_int("SKINTOKENS_DEFAULT_NUM_BEAMS", 1, 1, 8)
DEFAULT_NUM_SAMPLES = env_int("SKINTOKENS_DEFAULT_NUM_SAMPLES", 1, 1, 8)
MAX_NEW_TOKENS = env_int("SKINTOKENS_MAX_NEW_TOKENS", 2048, 128, 4096)
# Uploads are untrusted: capped in size here and parsed with resource limits
# (SKINTOKENS_INPUT_* env vars, see src/rig_package/glb_io/limits.py), never
# allowed to reference files outside the upload itself.
MAX_UPLOAD_MB = upload_limit_mb()
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
USE_GENERATION_CACHE = env_flag("SKINTOKENS_USE_CACHE", True)
EMPTY_CACHE_AFTER_INFERENCE = env_flag("SKINTOKENS_EMPTY_CACHE_AFTER_INFERENCE", True)
# Fast decode: SDPA attention + static KV cache + CUDA-graph capture gives ~2x
# on the autoregressive sampling stage, which dominates wall time. It requires
# ``num_beams == 1`` (a static cache cannot do beam reordering); beam requests
# transparently fall back to the original flash-attention + dynamic-cache path.
FAST_DECODE = env_flag("SKINTOKENS_FAST_DECODE", False)
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
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = Field(default=DEFAULT_NUM_BEAMS, ge=1, le=8)
    num_samples: int = Field(default=DEFAULT_NUM_SAMPLES, ge=1, le=8)
    seed: int | None = None
