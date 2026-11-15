"""Decode attention over a static cache matches plain masked attention."""
import pytest
import torch

from src.model import decode_attention as da


def reference(query, key, value, valid, scaling):
    rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, dim=1).float()
    v = value.repeat_interleave(rep, dim=1).float()
    scores = (query.float() @ k.transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    return (torch.softmax(scores, -1) @ v).to(query.dtype)


def make_case(device, dtype, *, b=2, hq=16, hkv=8, s=3192, d=128, filled=(600, 1), gaps=False,
              broadcast=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(b, hq, 1, d, generator=g).to(device, dtype)
    key = torch.randn(b, hkv, s, d, generator=g).to(device, dtype)
    value = torch.randn(b, hkv, s, d, generator=g).to(device, dtype)
    valid = torch.zeros(b, s, dtype=torch.bool)
    for row, n in zip(range(b), filled * b):
        valid[row, :n] = True
    if gaps:  # left padding / holes inside the filled part, incl. a whole empty block
        valid[:, 3:7] = False
        valid[:, 64:128] = False
    if broadcast:
        valid = valid[:1]
    # The garbage in unfilled slots must not leak into the result.
    key[..., s - 1, :] = 1e4
    mask4d = valid[:, None, None, :].to(device)
    return query, key, value, mask4d, valid.to(device).expand(b, s), d ** -0.5


CASES = [
    dict(),
    dict(filled=(1,)),
    dict(filled=(3192,)),
    dict(filled=(130, 1500)),
    dict(gaps=True),
    dict(broadcast=True, filled=(777,)),
    dict(b=1, s=257, filled=(200,)),
]


@pytest.mark.parametrize("case", CASES)
def test_math_matches_reference(case):
    query, key, value, mask, valid, scale = make_case("cpu", torch.float32, **case)
    out = da.decode_attention_math(query, key, value, mask, scale)
    torch.testing.assert_close(out, reference(query, key, value, valid, scale), atol=1e-5, rtol=1e-4)


def test_math_accepts_additive_float_mask():
    query, key, value, mask, valid, scale = make_case("cpu", torch.float32, filled=(300,))
    additive = torch.zeros(mask.shape).masked_fill(~mask, torch.finfo(torch.float32).min)
    out = da.decode_attention_math(query, key, value, additive, scale)
    torch.testing.assert_close(out, reference(query, key, value, valid, scale), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_math_preserves_float32_accumulation_under_autocast(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # pipeline.py enables TF32 globally; isolate the autocast regression
    # from shape-dependent TF32 rounding in the independent reference.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        query, key, value, mask, valid, scale = make_case(
            device, torch.float32, s=257, filled=(1, 200))
        expected = reference(query, key, value, valid, scale)
        with torch.autocast(device, dtype=torch.bfloat16):
            out = da.decode_attention_math(query, key, value, mask, scale)
        assert out.dtype == query.dtype
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


needs_triton = pytest.mark.skipif(not da._has_triton(), reason="needs CUDA and Triton")


@needs_triton
@pytest.mark.parametrize("case", CASES)
def test_triton_matches_reference(case):
    query, key, value, mask, valid, scale = make_case("cuda", torch.bfloat16, **case)
    out = da.decode_attention_triton(query, key, value, mask, scale)
    torch.testing.assert_close(out, reference(query, key, value, valid, scale), atol=2e-2, rtol=2e-2)


@needs_triton
def test_triton_inside_compiled_cuda_graph():
    """The op compiles with inductor and replays from a CUDA graph."""
    query, key, value, mask, valid, scale = make_case("cuda", torch.bfloat16, filled=(400,))

    @torch.compile(backend="inductor", mode="reduce-overhead")
    def step(q, k, v, m):
        return da.decode_attention_triton(q, k, v, m, scale) * 1.0

    for _ in range(3):  # warmup, record, replay
        out = step(query, key, value, mask)
    torch.testing.assert_close(out, reference(query, key, value, valid, scale), atol=2e-2, rtol=2e-2)
    # The filled length changes between replays without re-capturing.
    mask[:, :, :, 400:900] = True
    valid = mask[:, 0, 0, :].expand(query.shape[0], -1)
    out = step(query, key, value, mask)
    torch.testing.assert_close(out, reference(query, key, value, valid, scale), atol=2e-2, rtol=2e-2)
