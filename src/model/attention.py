"""Attention backend selection shared by the LLM and the VAE/encoder blocks.

flash-attn is optional: it often has no prebuilt wheel for new torch releases
(e.g. torch 2.11 on Windows, or the slim ``torch-runtime`` Docker base). When it
is missing we fall back to PyTorch SDPA, which on CUDA dispatches to its own
fused flash / memory-efficient kernels.
"""

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FLASH_ATTN_IMPL = "flash_attention_2"
SDPA_ATTN_IMPL = "sdpa"


def _sdpa_flash_layout(q, k, v, *args, **kwargs):
    """flash-attn compatible signature on top of SDPA.

    Inputs/outputs use the flash-attn layout ``(B, L, H, D)``; SDPA expects
    ``(B, H, L, D)``. Supports ``causal``/``softmax_scale`` and GQA.
    """
    causal = kwargs.get("causal", False)
    scale = kwargs.get("softmax_scale", None)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return out.transpose(1, 2), None


try:  # FlashAttention-3 (Hopper)
    from flash_attn_interface import flash_attn_func  # type: ignore
    FLASH_ATTN_BACKEND = "flash_attn_3"
except Exception:
    try:  # FlashAttention-2
        from flash_attn.flash_attn_interface import flash_attn_func as _fa2_func  # type: ignore

        def flash_attn_func(*args, **kwargs):
            return _fa2_func(*args, **kwargs), None

        FLASH_ATTN_BACKEND = "flash_attn_2"
    except Exception:
        flash_attn_func = _sdpa_flash_layout
        FLASH_ATTN_BACKEND = "sdpa"


def resolve_llm_attn_implementation() -> str:
    """Attention implementation for the HF transformer.

    ``SKINTOKENS_ATTN_IMPL`` overrides it (e.g. ``sdpa``, ``eager``,
    ``flash_attention_2``); otherwise flash_attention_2 is used when the
    flash-attn package is usable, else SDPA.
    """
    override = os.environ.get("SKINTOKENS_ATTN_IMPL", "").strip()
    if override:
        return override
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return FLASH_ATTN_IMPL
    except Exception:
        pass
    return SDPA_ATTN_IMPL


LLM_ATTN_IMPL = resolve_llm_attn_implementation()
logger.info(
    "Attention backends: llm=%s, vae/encoder=%s, torch=%s",
    LLM_ATTN_IMPL, FLASH_ATTN_BACKEND, torch.__version__,
)
