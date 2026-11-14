"""Request seeding: every RNG a request touches is seeded and restored."""
import random
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.sampler import SamplerResult
from src.data.seeding import derive_seed, sample_point_clouds, seeded_rng
from src.server.config import MAX_SEED, GenParams


def draw():
    return np.random.rand(4).tolist(), random.random(), torch.rand(4).tolist()


def test_seeded_rng_is_deterministic():
    with seeded_rng(123):
        a = draw()
    with seeded_rng(123):
        b = draw()
    assert a == b
    with seeded_rng(124):
        assert draw() != a


def test_seeded_rng_restores_surrounding_streams():
    """A seeded request must not shift the next request's global RNG streams."""
    np.random.seed(999); random.seed(999); torch.manual_seed(999)
    ref = [draw(), draw()]

    np.random.seed(999); random.seed(999); torch.manual_seed(999)
    first = draw()
    with seeded_rng(123):
        draw()  # consumes the seeded streams; rolled back on exit
    assert [first, draw()] == ref


def test_seeded_rng_none_is_passthrough():
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    ref = draw()
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    with seeded_rng(None):
        assert draw() == ref


@pytest.mark.parametrize("seed", [0, 1, 2**32, MAX_SEED])
def test_large_seeds_are_accepted(seed):
    with seeded_rng(seed):
        draw()
    assert 0 <= derive_seed(seed, 7) <= MAX_SEED


def test_derived_seeds_are_stable_and_distinct():
    seeds = [derive_seed(5, i) for i in range(8)]
    assert seeds == [derive_seed(5, i) for i in range(8)]
    assert len(set(seeds)) == 8
    assert derive_seed(6, 0) not in seeds


class NumpySampler:
    """Draws from the global numpy RNG, like the real mesh sampler."""

    def sample(self, asset):
        pts = np.random.rand(16, 3)
        return SamplerResult(sampled_vertices=pts, sampled_normals=-pts)


def test_point_clouds_reproduce_per_seed_and_differ_per_sample():
    v1, n1 = sample_point_clouds(None, NumpySampler(), seed=3, num_samples=3)
    v2, _ = sample_point_clouds(None, NumpySampler(), seed=3, num_samples=3)
    assert v1.shape == n1.shape == (3, 16, 3) and v1.dtype == torch.float32
    assert torch.equal(v1, v2)
    assert not torch.equal(v1[0], v1[1])
    # Sample i does not depend on how many samples were requested.
    v_one, _ = sample_point_clouds(None, NumpySampler(), seed=3, num_samples=1)
    assert torch.equal(v_one[0], v1[0])
    v_other, _ = sample_point_clouds(None, NumpySampler(), seed=4, num_samples=1)
    assert not torch.equal(v_other[0], v1[0])


def test_gen_params_expose_only_seed_and_samples():
    assert set(GenParams.model_fields) == {"num_samples", "seed"}
    assert GenParams(seed=9).with_seed().seed == 9
    assert GenParams().with_seed().seed is not None
    with pytest.raises(ValueError):
        GenParams(seed=-1)


@pytest.mark.parametrize("configured, expected_max", [
    (None, 1), ("", 1), ("invalid", 1), ("0", 1), ("3", 3), ("8", 8), ("9", 8),
])
def test_gen_params_enforce_configured_sample_limit(monkeypatch, configured, expected_max):
    monkeypatch.delenv("SKINTOKENS_MAX_NUM_SAMPLES", raising=False)
    if configured is not None:
        monkeypatch.setenv("SKINTOKENS_MAX_NUM_SAMPLES", configured)
    # The retired default must no longer affect requests or their limit.
    monkeypatch.setenv("SKINTOKENS_DEFAULT_NUM_SAMPLES", "8")
    config = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/server/config.py"))
    params = config["GenParams"]
    assert config["MAX_NUM_SAMPLES"] == expected_max
    assert params().num_samples == 1
    assert params(num_samples=expected_max).num_samples == expected_max
    for invalid in (0, expected_max + 1):
        with pytest.raises(ValueError):
            params(num_samples=invalid)
