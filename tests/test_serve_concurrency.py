"""Concurrency tests for the multi-process serve pipeline.

Covered guarantees:
  * requests run in parallel on different worker processes (no global I/O or
    GPU lock), up to the pool size; the rest queue in FIFO order.
  * results never cross-contaminate between concurrent requests.
  * cancellation removes a queued request and stops a running one, freeing its
    worker.
  * a worker crash fails only its request and the worker is replaced.
"""

import asyncio
import time

import pytest

import serve
from conftest import run_with_pool
from src.server.pool import WorkerCrashed


def make_params(**kw):
    return serve.GenParams(**kw)


def sleep_job(seconds, tag="t"):
    return {"action": "sleep", "seconds": seconds, "request_id": tag}


def _span(result):
    pid, span = result
    start, end = eval(span.decode())
    return int(pid), start, end


# --------------------------------------------------------------------------- #
# Parallelism and isolation
# --------------------------------------------------------------------------- #
def test_requests_run_in_parallel_across_workers():
    async def scenario(pool):
        started = time.monotonic()
        results = await asyncio.gather(*[pool.submit(sleep_job(0.6, f"r{i}"), b"")
                                         for i in range(4)])
        return time.monotonic() - started, [_span(r) for r in results]

    elapsed, spans = run_with_pool(scenario, size=2)
    # Two workers, four 0.6 s jobs: two rounds, not four.
    assert elapsed < 4 * 0.6, elapsed
    assert len({pid for pid, _, _ in spans}) == 2
    # Every moment at most two jobs run, and some do run together.
    events = sorted([(s, 1) for _, s, _ in spans] + [(e, -1) for _, _, e in spans])
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    assert peak == 2


def test_no_cross_contamination_between_concurrent_requests():
    async def scenario(pool):
        return await asyncio.gather(*[
            serve._generate(f"data{i}".encode(), "m.glb", make_params(), f"req{i}")
            for i in range(6)
        ])

    results = run_with_pool(scenario, size=3)
    for i, samples in enumerate(results):
        glb, _meta = samples[0]
        assert glb == f"glb:req{i}:data{i}".encode()
    assert serve.state.active_jobs == 0


def test_progress_reaches_the_request_callback():
    seen = []

    async def scenario(pool):
        await serve._generate(b"x", "m.glb", make_params(), "req",
                              lambda p, s, e: seen.append((p, s)))

    run_with_pool(scenario, size=1)
    assert seen == [(50, "half way")]


def test_waiting_request_gets_queue_heartbeats_in_fifo_order():
    beats = []

    async def scenario(pool):
        async def queue_cb(phase, waited, ahead):
            beats.append((phase, ahead))

        first = asyncio.create_task(pool.submit(sleep_job(1.6), b""))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(pool.submit(sleep_job(0.01), b"", queue_callback=queue_cb))
        await asyncio.gather(first, second)
        return pool.status()

    status = run_with_pool(scenario, size=1)
    assert beats and beats[0] == ("worker", 0)
    assert status["busy"] == 0 and status["queued"] == 0


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
def test_cancel_running_request_frees_its_worker():
    async def scenario(pool):
        cancel = serve.CancellationToken()
        task = asyncio.create_task(pool.submit(sleep_job(30), b"", cancelled=cancel.event))
        await asyncio.sleep(0.3)
        started = time.monotonic()
        cancel.cancel("test cancel")
        with pytest.raises(serve.GenerationCancelled):
            await task
        stopped_after = time.monotonic() - started
        # The same (only) worker takes the next request.
        await pool.submit(sleep_job(0.01), b"")
        return stopped_after

    stopped_after = run_with_pool(scenario, size=1)
    assert stopped_after < 2.0


def test_cancel_queued_request_leaves_the_queue():
    async def scenario(pool):
        cancel = serve.CancellationToken()
        first = asyncio.create_task(pool.submit(sleep_job(0.8), b""))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(pool.submit(sleep_job(0.01), b"", cancelled=cancel.event))
        await asyncio.sleep(0.05)
        assert pool.status()["queued"] == 1
        cancel.cancel("test cancel")
        with pytest.raises(serve.GenerationCancelled):
            await queued
        assert pool.status()["queued"] == 0
        await first
        return pool.status()

    status = run_with_pool(scenario, size=1)
    assert status["idle"] == 1


def test_generate_cancellation_raises_with_reason():
    async def scenario(pool):
        cancel = serve.CancellationToken()
        busy = asyncio.create_task(pool.submit(sleep_job(0.5), b""))
        await asyncio.sleep(0.05)
        task = asyncio.create_task(serve._generate(b"d", "m.glb", make_params(), "req", None, cancel))
        await asyncio.sleep(0.05)
        assert serve.state.active_jobs == 1
        cancel.cancel("client requested cancellation")
        with pytest.raises(serve.GenerationCancelled, match="client requested"):
            await task
        await busy

    run_with_pool(scenario, size=1)
    assert serve.state.active_jobs == 0


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def test_worker_exception_reaches_the_request():
    async def scenario(pool):
        with pytest.raises(ValueError, match="bad request value"):
            await pool.submit({"action": "raise"}, b"")
        # The worker survives an exception in a job.
        return await pool.submit({"action": "echo"}, b"ok")

    assert run_with_pool(scenario, size=1)[0] == b"glb:-:ok"


def test_crashed_worker_fails_its_request_and_is_replaced():
    async def scenario(pool):
        with pytest.raises(WorkerCrashed):
            await pool.submit({"action": "crash"}, b"")
        result = await asyncio.wait_for(pool.submit({"action": "echo"}, b"after"), 30)
        return result, pool.status()

    result, status = run_with_pool(scenario, size=1, restart_delay=0.1)
    assert result[0] == b"glb:-:after"
    assert status["restarts"] == 1


def test_unsupported_extension_rejected():
    async def scenario():
        with pytest.raises(ValueError):
            await serve._generate(b"d", "model.txt", make_params(), "req")

    asyncio.run(scenario())
    assert serve.state.active_jobs == 0
