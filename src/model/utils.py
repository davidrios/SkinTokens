import threading
from typing import Dict, Tuple

import torch


def _fps_loop(points: torch.Tensor, selected: torch.Tensor, farthest: torch.Tensor,
              distances: torch.Tensor) -> None:
    """Farthest point sampling of ``selected.shape[0]`` of *points* into
    *selected*, starting at *farthest* (a 0-dim index). In place and with no
    host round-trip, so it can be captured as a CUDA graph."""
    distances.fill_(float("inf"))
    for i in range(selected.shape[0]):
        selected[i] = farthest
        centroid = points.index_select(0, farthest.view(1))
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        torch.minimum(distances, dist, out=distances)
        farthest.copy_(torch.argmax(distances))


class _FpsGraph:
    """One sampling size's loop as a CUDA graph: hundreds of iterations of
    tiny kernels, which eagerly are bound by their launches (and, beside
    another busy thread, by taking the GIL back after each one)."""

    def __init__(self, n: int, channels: int, k: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        self.points = torch.zeros((n, channels), dtype=dtype, device=device)
        self.selected = torch.zeros(k, dtype=torch.long, device=device)
        self.farthest = torch.zeros((), dtype=torch.long, device=device)
        self.distances = torch.empty(n, dtype=torch.float32, device=device)
        # Where the last replay ran: the next one, maybe on another stream,
        # must not overwrite the buffers before it has finished with them.
        self.done = None
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            self._run()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        # thread_local: other threads (the batched decoder) keep using the GPU.
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self._run()

    def _run(self) -> None:
        self.farthest.zero_()
        _fps_loop(self.points, self.selected, self.farthest, self.distances)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(points.device)
        if self.done is not None:
            stream.wait_event(self.done)
        self.points.copy_(points)
        self.graph.replay()
        out = self.selected.clone()
        self.done = torch.cuda.Event()
        self.done.record(stream)
        return out


_FPS_GRAPHS: Dict[Tuple, _FpsGraph] = {}
_FPS_LOCK = threading.Lock()


def _fps_indices(points: torch.Tensor, k: int, random_start: bool) -> torch.Tensor:
    """Local indices of *k* farthest-point samples of *points* ``(n, C)``."""
    n = points.shape[0]
    if points.is_cuda and not random_start and not torch.cuda.is_current_stream_capturing():
        key = (n, points.shape[1], k, points.dtype, points.device,
               torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        with _FPS_LOCK:
            graph = _FPS_GRAPHS.get(key)
            if graph is None:
                graph = _FPS_GRAPHS[key] = _FpsGraph(n, points.shape[1], k, points.dtype,
                                                     points.device)
            return graph(points)
    selected = torch.empty(k, dtype=torch.long, device=points.device)
    if random_start:
        farthest = torch.randint(n, (1,), device=points.device).reshape(())
    else:
        farthest = torch.zeros((), dtype=torch.long, device=points.device)
    distances = torch.empty(n, dtype=torch.float32, device=points.device)
    _fps_loop(points, selected, farthest, distances)
    return selected


def fps(
    x: torch.Tensor,
    batch: torch.Tensor,
    ratio: float,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: (N, C) points.
        batch: (N,) batch indices for each point.
        ratio: sampling ratio in (0, 1].
        random_start: whether to start from a random point per batch.
    
    Returns:
        1D tensor of sampled indices in the flattened input space.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (N, C), got {tuple(x.shape)}")
    if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
        raise ValueError("batch must be 1D and aligned with x")
    if not (0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    sampled_indices = []
    unique_batches = torch.unique(batch)

    for batch_id in unique_batches:
        mask = batch == batch_id
        points = x[mask]
        num_points = points.shape[0]

        if num_points == 0:
            continue

        num_samples = max(1, int(round(num_points * ratio)))
        num_samples = min(num_samples, num_points)

        selected_local = _fps_indices(points, num_samples, random_start)
        global_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)[selected_local]
        sampled_indices.append(global_indices)

    if not sampled_indices:
        return torch.empty((0,), dtype=torch.long, device=x.device)
    return torch.cat(sampled_indices, dim=0)


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    reduce: str = "sum",
) -> torch.Tensor:
    """
    Args:
        src: source tensor with shape (N, ...).
        indptr: CSR index pointer with shape (S + 1,).
        reduce: one of {"sum", "mean", "min", "max"}.
    
    Returns:
        Reduced tensor with shape (S, ...).
    """
    if src.ndim < 1:
        raise ValueError(f"Expected src to have at least 1 dim, got {src.ndim}")
    if indptr.ndim != 1:
        raise ValueError(f"Expected indptr to be 1D, got shape {tuple(indptr.shape)}")
    if indptr.numel() < 1:
        raise ValueError("indptr must contain at least one element")
    if reduce not in {"sum", "mean", "min", "max"}:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    indptr = indptr.to(device=src.device, dtype=torch.long)
    segments = indptr.numel() - 1
    out_shape = (segments, *src.shape[1:])

    if reduce in {"sum", "mean"}:
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    elif reduce == "min":
        out = torch.full(out_shape, float("inf"), dtype=src.dtype, device=src.device)
    else:
        out = torch.full(out_shape, float("-inf"), dtype=src.dtype, device=src.device)

    for i in range(segments):
        start = indptr[i].item()
        end = indptr[i + 1].item()
        if end <= start:
            continue

        chunk = src[start:end]
        if reduce == "sum":
            out[i] = chunk.sum(dim=0)
        elif reduce == "mean":
            out[i] = chunk.mean(dim=0)
        elif reduce == "min":
            out[i] = chunk.min(dim=0).values
        else:
            out[i] = chunk.max(dim=0).values

    if reduce == "min":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    elif reduce == "max":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)

    return out