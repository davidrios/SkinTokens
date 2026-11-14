"""Skin weights from a sampled point cloud onto the mesh vertices.

The model predicts skin at the points it was given (a few tens of thousands);
the mesh may have far more vertices. Each vertex takes the inverse-distance
weighted mean of its ``k`` nearest points' weights.

:func:`interpolate_skin` returns the dense ``(N, J)`` matrix. For export, which
keeps only each vertex's heaviest few influences, :func:`interpolate_top_k`
returns just those, without materializing the dense matrix (on the GPU when
one is given, else in CPU threads).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial import cKDTree  # type: ignore

from .utils import cpu_workers, parallel_rows

NEIGHBORS = 8


def nearest(vertices: ndarray, sampled_vertices: ndarray, k: int = NEIGHBORS) -> Tuple[ndarray, ndarray]:
    """``(indices, distances)``, each ``(N, k)``: every vertex's ``k`` nearest sampled points."""
    k = min(k, len(sampled_vertices))
    distances, indices = cKDTree(sampled_vertices).query(vertices, k=k, workers=cpu_workers(16))
    if k == 1:
        return indices[:, None], distances[:, None]
    return indices, distances


def idw(distances):
    """Normalized inverse-distance weights (numpy or torch)."""
    weights = 1.0 / (distances + 1e-8)
    return weights / weights.sum(axis=1, keepdims=True) if isinstance(weights, ndarray) \
        else weights / weights.sum(dim=1, keepdim=True)


def knn_weights(vertices: ndarray, sampled_vertices: ndarray, k: int = NEIGHBORS) -> Tuple[ndarray, ndarray]:
    """``(indices, weights)``, each ``(N, k)``: every vertex's ``k`` nearest
    sampled points and their normalized inverse-distance weights (float64)."""
    indices, distances = nearest(vertices, sampled_vertices, k)
    return indices, idw(distances)


def interpolate_skin(vertices: ndarray, sampled_vertices: ndarray, sampled_skin: ndarray,
                     k: int = NEIGHBORS) -> ndarray:
    """The dense ``(N, J)`` skin at ``vertices``."""
    indices, weights = knn_weights(vertices, sampled_vertices, k)
    if indices.shape[1] == 1:
        return sampled_skin[indices[:, 0]]
    out = np.empty((len(vertices), sampled_skin.shape[1]),
                   dtype=np.result_type(weights.dtype, sampled_skin.dtype))

    def rows(r: slice):
        out[r] = np.einsum("nk,nkj->nj", weights[r], sampled_skin[indices[r]])

    parallel_rows(rows, len(vertices), min_chunk=16384)
    return out


def select_top_k(skin: ndarray, k: int) -> Tuple[ndarray, ndarray]:
    """Per row, the ``k`` largest entries of a cleaned (finite, non-negative)
    ``skin``: ``(ids, weights)``, heaviest first, lowest index on ties."""
    J = skin.shape[1]
    if k < J:
        # Select the k largest per row without sorting all J columns, then order
        # just those.
        part = np.sort(np.argpartition(-skin, k - 1, axis=1)[:, :k], axis=1)
        w = np.take_along_axis(skin, part, axis=1)
        order = np.argsort(-w, axis=1, kind='stable')
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(w, order, axis=1)
    ids = np.argsort(-skin, axis=1, kind='stable')
    return ids, np.take_along_axis(skin, ids, axis=1)


def clean_skin(skin: ndarray) -> ndarray:
    """Non-finite entries to zero (inf to the float max, as ``nan_to_num``), negatives to zero."""
    return np.maximum(np.nan_to_num(skin), 0.)


def interpolate_top_k(vertices: ndarray, sampled_vertices: ndarray, sampled_skin: ndarray,
                      top_k: int, k: int = NEIGHBORS, device: Optional[str] = None,
                      chunk: int = 65536) -> Tuple[ndarray, ndarray]:
    """The ``top_k`` heaviest influences of every vertex of the interpolated skin.

    Equals ``select_top_k(clean_skin(interpolate_skin(...)), top_k)`` (the
    GPU path up to float rounding, and which of exactly tied weights is kept).
    Returns ``(ids int64, weights float64)``, each ``(N, min(top_k, J))``.
    ``device`` is a torch CUDA device to run on, or ``None``/``"cpu"`` for CPU threads.
    """
    J = sampled_skin.shape[1]
    top_k = J if top_k is None or top_k <= 0 else min(top_k, J)
    indices, distances = nearest(vertices, sampled_vertices, k)
    if device is not None and str(device).startswith("cuda"):
        return _top_k_torch(indices, distances, sampled_skin, top_k, device, chunk)

    weights = idw(distances)
    n = len(vertices)
    ids = np.empty((n, top_k), dtype=np.int64)
    w = np.empty((n, top_k), dtype=np.float64)

    def rows(r: slice):
        if indices.shape[1] == 1:
            dense = sampled_skin[indices[r, 0]].astype(np.float64)
        else:
            dense = np.einsum("nk,nkj->nj", weights[r], sampled_skin[indices[r]])
        ids[r], w[r] = select_top_k(clean_skin(dense), top_k)

    parallel_rows(rows, n, min_chunk=16384)
    return ids, w


def _top_k_torch(indices, distances, sampled_skin, top_k, device, chunk):
    import torch
    skin = torch.as_tensor(np.ascontiguousarray(sampled_skin), device=device).double()
    idx = torch.as_tensor(indices, device=device)
    wts = idw(torch.as_tensor(distances, device=device, dtype=torch.float64))
    n = len(indices)
    ids_out = torch.empty((n, top_k), dtype=torch.int64, device=device)
    w_out = torch.empty((n, top_k), dtype=torch.float64, device=device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        i, w = idx[lo:hi], wts[lo:hi]
        dense = w[:, 0, None] * skin[i[:, 0]]
        for j in range(1, i.shape[1]):
            dense.addcmul_(w[:, j, None], skin[i[:, j]])
        dense = torch.nan_to_num(dense).clamp_min_(0.)
        val, sel = torch.topk(dense, top_k, dim=1, sorted=False)
        # heaviest first, lowest joint index on ties (as select_top_k)
        sel, order = torch.sort(sel, dim=1)
        val = torch.gather(val, 1, order)
        val, order = torch.sort(val, dim=1, descending=True, stable=True)
        ids_out[lo:hi] = torch.gather(sel, 1, order)
        w_out[lo:hi] = val
    return ids_out.cpu().numpy(), w_out.cpu().numpy()
