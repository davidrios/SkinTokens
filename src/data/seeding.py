"""Request-scoped seeding for reproducible inference.

Decoding is greedy, so the model adds no randomness of its own. What is left is
where points are sampled on the input mesh (numpy and python ``random``) and a
few subsamples taken during export. Seeding these from the request's seed makes
a request reproducible: sample ``i`` draws its point cloud, and is exported,
under ``derive_seed(seed, i)``, so ``num_samples`` clouds differ from each other
but the same ``(seed, num_samples)`` always gives the same outputs.
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

MAX_SEED = 2**63 - 1


def enable_deterministic_kernels() -> None:
    """Make every process compute bit-identical results from the same inputs.

    Without this, each process's torch.compile (inductor) build of the decode
    graph lands on one of a few numerically different kernel sets, so a seed
    reproduces within a process but differs between processes (e.g. between
    server workers or restarts). Inductor's own ``deterministic`` flag is not
    enough; torch's deterministic-algorithms mode is. It costs decode speed
    (measured on the fast path: 2.1 → 3.3 ms/token at batch 1).

    Must run before the first compilation: dynamo guards on this mode, so
    changing it later recompiles. ``warn_only`` keeps an op without a
    deterministic implementation from failing a request, and uninitialized
    memory is not filled (it does not affect the results here).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False


def derive_seed(seed: int, index: int) -> int:
    """An independent seed for sample ``index`` of a request seeded ``seed``."""
    return int(np.random.SeedSequence([seed, index]).generate_state(1, np.uint64)[0] & MAX_SEED)


@contextmanager
def seeded_rng(seed: int | None):
    """Seed numpy, python ``random`` and torch (CPU and CUDA), restoring them on exit.

    Mesh sampling and transformers use these *global* generators, so the seed
    has to be applied globally; a worker process runs one request at a time, so
    this is race-free. Restoring the state keeps a seeded request from shifting
    the streams of whatever runs next. ``None`` leaves everything untouched.
    """
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    py_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed((seed & 0xFFFFFFFF) ^ (seed >> 32))  # numpy's legacy seed is 32-bit
        random.seed(seed)
        torch.manual_seed(seed)  # also seeds every CUDA device
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def sample_point_clouds(asset, sampler, seed: int, num_samples: int) -> Tuple[Tensor, Tensor]:
    """Draw ``num_samples`` point clouds from ``asset`` (already transformed),
    cloud ``i`` under ``derive_seed(seed, i)``.

    Returns ``(vertices, normals)``, each ``(num_samples, P, 3)`` float32.
    """
    vertices, normals = [], []
    for i in range(num_samples):
        with seeded_rng(derive_seed(seed, i)):
            res = sampler.sample(asset=asset)
        if res.sampled_vertices is None or res.sampled_normals is None:
            raise ValueError("the sampler produced no points or normals")
        vertices.append(torch.from_numpy(np.ascontiguousarray(res.sampled_vertices)).float())
        normals.append(torch.from_numpy(np.ascontiguousarray(res.sampled_normals)).float())
    return torch.stack(vertices), torch.stack(normals)
