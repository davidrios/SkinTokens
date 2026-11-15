"""FPS inference parity for the deployed float32 and bfloat16 inputs.

Training/autograd and float64 parity are outside this suite's scope. On these
fixed inputs, CUDA graph capture and replay should preserve the eager indices.
"""
import pytest
import torch

from src.model import utils


def eager_fps(x, batch, ratio):
    """The original loop, one host round-trip per sample."""
    out = []
    for batch_id in torch.unique(batch):
        mask = batch == batch_id
        points = x[mask]
        k = min(max(1, int(round(points.shape[0] * ratio))), points.shape[0])
        farthest = 0
        distances = torch.full((points.shape[0],), float("inf"), device=x.device)
        selected = torch.empty(k, dtype=torch.long, device=x.device)
        for i in range(k):
            selected[i] = farthest
            dist = torch.sum((points - points[farthest]) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()
        out.append(torch.nonzero(mask, as_tuple=False).squeeze(-1)[selected])
    return torch.cat(out)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@torch.inference_mode()
def test_fps_inference_picks_the_points_the_eager_loop_picks(device, dtype, autocast):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(3 * 256, 3, generator=generator).to(device, dtype)
    batch = torch.arange(3, device=device).repeat_interleave(256)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device == "cuda"):
        expected = eager_fps(x, batch, 0.25)
        for _ in range(2):  # captures, then replays
            assert torch.equal(utils.fps(x, batch, ratio=0.25), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_fps_inference_graph_is_safe_across_streams():
    x = torch.rand(512, 3, device="cuda")
    batch = torch.zeros(512, dtype=torch.long, device="cuda")
    expected = eager_fps(x, batch, 0.25)
    results = []
    for _ in range(4):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(utils.fps(x, batch, ratio=0.25))
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert all(torch.equal(r, expected) for r in results)
