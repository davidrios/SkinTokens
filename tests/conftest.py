"""Shared fixtures for the server tests.

The pool tests start real worker processes (``spawn``) running the model-free
spec in :mod:`pool_fixtures`, so they exercise the real gateway ↔ worker
protocol (dispatch, progress, cancellation, crashes) without a GPU.
"""

import asyncio
import logging

import pytest

import serve

FIXTURE_SPEC = "pool_fixtures:ECHO"


@pytest.fixture(autouse=True)
def _quiet_logs():
    """Silence the per-request INFO spam during tests."""
    previous = serve.logger.level
    serve.logger.setLevel(logging.WARNING)
    yield
    serve.logger.setLevel(previous)


@pytest.fixture(autouse=True)
def fresh_state():
    serve.state.pool = None
    serve.state.cpu_pool = None
    serve.state.active_jobs = 0
    serve.state.admitted = 0
    serve.state.ready = True
    yield
    serve.state.pool = None
    serve.state.cpu_pool = None


def run_with_pool(scenario, size=2, **pool_kwargs):
    """Run ``scenario(pool)`` on a started fixture pool, installed as the
    server's pool, and close it afterwards."""

    async def main():
        pool = serve.WorkerPool(FIXTURE_SPEC, size, **pool_kwargs)
        await pool.start()
        serve.state.pool = pool
        try:
            return await scenario(pool)
        finally:
            serve.state.pool = None
            await pool.close(timeout=5)

    return asyncio.run(main())


def run_with_stages(scenario, gpu_workers=1, cpu_workers=2):
    """Run ``scenario(gpu_pool, cpu_pool)`` on started fixture pools running
    stand-in stages (``pool_fixtures.STAGED_*``), installed as the server's
    pools, and close them afterwards."""

    async def main():
        gpu = serve.WorkerPool("pool_fixtures:STAGED_GPU", gpu_workers, label="gpu-worker")
        cpu = serve.WorkerPool("pool_fixtures:STAGED_CPU", cpu_workers, cuda=False,
                               label="cpu-worker")
        await asyncio.gather(gpu.start(), cpu.start())
        serve.state.pool, serve.state.cpu_pool = gpu, cpu
        try:
            return await scenario(gpu, cpu)
        finally:
            serve.state.pool = serve.state.cpu_pool = None
            await asyncio.gather(gpu.close(timeout=5), cpu.close(timeout=5))

    return asyncio.run(main())
