"""Single-token (decode) attention over a pre-allocated static KV cache.

The fast decode path (static cache + CUDA graphs) needs fixed shapes, so every
step's keys/values are the whole pre-allocated cache (prefix + max_new_tokens
slots, ~3.2k) and a mask marks the slots filled so far. Plain SDPA then runs
its memory-efficient kernel over every slot, filled or not: that was ~60% of
the GPU time per token.

This module registers an attention implementation, ``skintokens_decode``,
with transformers. Prefill (more than one query token) still goes to HF's SDPA
attention; single-token decode steps use, by ``SKINTOKENS_DECODE_ATTN``:

``triton`` (default where Triton is available)
    A split-KV decode kernel. Each program covers one KV head (all of its
    grouped query heads) over one slice of the cache and returns early on mask
    blocks with no valid slot, so the work follows the *filled* length while
    the launch shape stays fixed (CUDA-graph safe). Partial results are merged
    by a second small kernel (log-sum-exp combine).
``math``
    Grouped-query attention as plain tensor ops, which inductor fuses. Still
    reads the whole cache, but without the memory-efficient kernel and without
    materializing repeated K/V heads. The fallback without Triton (Windows).
``sdpa``
    The previous behaviour: HF's SDPA attention for every step.
"""
from __future__ import annotations

import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)

DECODE_ATTN_IMPL = "skintokens_decode"
_MODES = ("triton", "math", "sdpa")


def _has_triton() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def _resolve_mode() -> str:
    mode = os.environ.get("SKINTOKENS_DECODE_ATTN", "").strip().lower()
    if not mode:
        return "triton" if _has_triton() else "math"
    if mode not in _MODES:
        logger.warning("Unknown SKINTOKENS_DECODE_ATTN=%r; using math", mode)
        return "math"
    if mode == "triton" and not _has_triton():
        logger.warning("SKINTOKENS_DECODE_ATTN=triton but Triton/CUDA is unavailable; using math")
        return "math"
    return mode


DECODE_ATTN_MODE = _resolve_mode()


def _valid_slots(attention_mask: torch.Tensor) -> torch.Tensor:
    """``(B or 1, S)`` bool: the key slots the (single) query may attend to."""
    mask = attention_mask[:, 0, -1, :]
    if mask.dtype != torch.bool:
        mask = mask == 0  # additive mask: 0 = attend, -inf/min = masked
    return mask


# --------------------------------------------------------------------------- #
# math (option 1)
# --------------------------------------------------------------------------- #
def decode_attention_math(query, key, value, attention_mask, scaling):
    """``query`` (B, Hq, 1, D); ``key``/``value`` (B, Hkv, S, D) → (B, Hq, 1, D)."""
    b, hq, _, d = query.shape
    hkv = key.shape[1]
    rep = hq // hkv
    q = query.reshape(b, hkv, rep, d).float()
    scores = torch.einsum("bhrd,bhsd->bhrs", q, key.float()) * scaling
    valid = _valid_slots(attention_mask)[:, None, None, :]
    scores = scores.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhrs,bhsd->bhrd", probs, value.float())
    return out.reshape(b, hq, 1, d).to(query.dtype)


# --------------------------------------------------------------------------- #
# Triton split-KV (option 2)
# --------------------------------------------------------------------------- #
SPLIT_SIZE = 32
BLOCK_N = 32

if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_split_kernel(
        Q, K, V, M, Out, Lse,
        sq_b, sq_h, sq_d,
        sk_b, sk_h, sk_s, sk_d,
        sv_b, sv_h, sv_s, sv_d,
        sm_b, sm_s,
        so_b, so_h, so_p, so_d,
        sl_b, sl_h, sl_p,
        S, scale,
        HKV: tl.constexpr, REP: tl.constexpr, REP_P2: tl.constexpr, D: tl.constexpr,
        BLOCK: tl.constexpr, SPLIT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        b = pid // HKV
        hkv = pid % HKV
        offs_r = tl.arange(0, REP_P2)
        r_ok = offs_r < REP
        offs_d = tl.arange(0, D)
        hq = hkv * REP + offs_r

        q = tl.load(Q + b * sq_b + hq[:, None] * sq_h + offs_d[None, :] * sq_d,
                    mask=r_ok[:, None], other=0.0).to(tl.float32) * scale
        m_i = tl.full([REP_P2], float("-inf"), tl.float32)
        l_i = tl.zeros([REP_P2], tl.float32)
        acc = tl.zeros([REP_P2, D], tl.float32)

        start = split * SPLIT
        for n0 in range(0, SPLIT, BLOCK):
            offs_n = start + n0 + tl.arange(0, BLOCK)
            valid = tl.load(M + b * sm_b + offs_n * sm_s, mask=offs_n < S, other=0) != 0
            # Blocks with no valid slot (the unfilled tail) cost one mask load.
            if tl.max(valid.to(tl.int32), axis=0) > 0:
                k = tl.load(K + b * sk_b + hkv * sk_h + offs_n[:, None] * sk_s + offs_d[None, :] * sk_d,
                            mask=valid[:, None], other=0.0).to(tl.float32)
                qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
                qk = tl.where(valid[None, :], qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
                v = tl.load(V + b * sv_b + hkv * sv_h + offs_n[:, None] * sv_s + offs_d[None, :] * sv_d,
                            mask=valid[:, None], other=0.0).to(tl.float32)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
                m_i = m_new

        has = l_i > 0
        out = acc / tl.where(has, l_i, 1.0)[:, None]
        lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
        tl.store(Out + b * so_b + hq[:, None] * so_h + split * so_p + offs_d[None, :] * so_d,
                 out, mask=r_ok[:, None])
        tl.store(Lse + b * sl_b + hq * sl_h + split * sl_p, lse, mask=r_ok)

    @triton.jit
    def _decode_combine_kernel(
        Part, Lse, Out,
        sp_b, sp_h, sp_p, sp_d,
        sl_b, sl_h, sl_p,
        so_b, so_h, so_d,
        NSPLIT, HQ: tl.constexpr, NSPLIT_P2: tl.constexpr, D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // HQ
        h = pid % HQ
        offs_p = tl.arange(0, NSPLIT_P2)
        p_ok = offs_p < NSPLIT
        offs_d = tl.arange(0, D)
        lse = tl.load(Lse + b * sl_b + h * sl_h + offs_p * sl_p, mask=p_ok, other=float("-inf"))
        m = tl.max(lse, axis=0)
        w = tl.exp(lse - m)
        part = tl.load(Part + b * sp_b + h * sp_h + offs_p[:, None] * sp_p + offs_d[None, :] * sp_d,
                       mask=p_ok[:, None], other=0.0)
        out = tl.sum(w[:, None] * part, axis=0) / tl.sum(w, axis=0)
        tl.store(Out + b * so_b + h * so_h + offs_d * so_d, out.to(Out.dtype.element_ty))

    @torch.library.custom_op("skintokens::decode_attention", mutates_args=())
    def _decode_attention_op(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             valid: torch.Tensor, scaling: float) -> torch.Tensor:
        b, hq, _, d = query.shape
        hkv, s = key.shape[1], key.shape[2]
        rep = hq // hkv
        nsplit = triton.cdiv(s, SPLIT_SIZE)
        # Reinterpret the bool mask (same byte size): no copy kernel per layer.
        valid = valid.view(torch.uint8) if valid.dtype == torch.bool else valid.to(torch.uint8)
        if valid.shape[0] != b:
            valid = valid.expand(b, s)
        part = torch.empty((b, hq, nsplit, d), device=query.device, dtype=torch.float32)
        lse = torch.empty((b, hq, nsplit), device=query.device, dtype=torch.float32)
        out = torch.empty((b, hq, 1, d), device=query.device, dtype=query.dtype)
        _decode_split_kernel[(b * hkv, nsplit)](
            query, key, value, valid, part, lse,
            query.stride(0), query.stride(1), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            valid.stride(0), valid.stride(1),
            part.stride(0), part.stride(1), part.stride(2), part.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            s, scaling,
            HKV=hkv, REP=rep, REP_P2=triton.next_power_of_2(rep), D=d,
            BLOCK=BLOCK_N, SPLIT=SPLIT_SIZE, num_warps=4,
        )
        _decode_combine_kernel[(b * hq,)](
            part, lse, out,
            part.stride(0), part.stride(1), part.stride(2), part.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            out.stride(0), out.stride(1), out.stride(3),
            nsplit, HQ=hq, NSPLIT_P2=triton.next_power_of_2(nsplit), D=d, num_warps=4,
        )
        return out

    @_decode_attention_op.register_fake
    def _(query, key, value, valid, scaling):
        return torch.empty_like(query)


def decode_attention_triton(query, key, value, attention_mask, scaling):
    """Same contract as :func:`decode_attention_math`, on the Triton kernels."""
    d = query.shape[-1]
    if d & (d - 1):
        raise ValueError(f"head_dim must be a power of two for the Triton kernel, got {d}")
    return torch.ops.skintokens.decode_attention(query, key, value,
                                                 _valid_slots(attention_mask), float(scaling))


# --------------------------------------------------------------------------- #
# transformers integration
# --------------------------------------------------------------------------- #
def skintokens_decode_attention(module, query, key, value, attention_mask, dropout=0.0,
                                scaling=None, **kwargs):
    """transformers attention function: prefill → SDPA, decode → ``DECODE_ATTN_MODE``."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if (query.shape[2] != 1 or attention_mask is None or DECODE_ATTN_MODE == "sdpa"
            or kwargs.get("sliding_window")):
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    if DECODE_ATTN_MODE == "triton":
        out = decode_attention_triton(query, key, value, attention_mask, scaling)
    else:
        out = decode_attention_math(query, key, value, attention_mask, scaling)
    return out.transpose(1, 2).contiguous(), None


_registered = False


def register() -> str:
    """Register ``skintokens_decode`` with transformers; returns its name."""
    global _registered
    if not _registered:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

        AttentionInterface.register(DECODE_ATTN_IMPL, skintokens_decode_attention)
        AttentionMaskInterface.register(DECODE_ATTN_IMPL, sdpa_mask)
        _registered = True
        logger.info("decode attention: %s", DECODE_ATTN_MODE)
    return DECODE_ATTN_IMPL
