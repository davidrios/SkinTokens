"""Workers that run several requests at once (continuous batching).

A worker whose spec has a ``serve`` loop and whose load reports ``slots``
receives up to that many jobs at a time (src/server/worker.py, pool.py).

Covered guarantees:
  * a worker runs up to its slots' jobs at once; further requests queue.
  * requests go to the least busy worker.
  * cancelling one job leaves the worker's other jobs running.
  * a crash fails every job the worker was running, and its replacement
    gets all its slots back.
"""

import asyncio
import time

import pytest

import serve
from src.server.pool import WorkerCrashed
from src.server.worker import GenerationCancelled


def sleep_job(seconds, tag="t"):
    return {"action": "sleep", "seconds": seconds, "request_id": tag}


def _span(result):
    pid, span = result
    start, end = eval(span.decode())
    return int(pid), start, end


def run_batched(scenario, size=1, slots=3, **kwargs):
    async def main():
        pool = serve.WorkerPool("pool_fixtures:BATCHED", size, options={"slots": slots}, **kwargs)
        await pool.start()
        try:
            return await scenario(pool)
        finally:
            await pool.close(timeout=5)

    return asyncio.run(main())


def test_a_worker_runs_its_slots_jobs_at_once_and_queues_the_rest():
    async def scenario(pool):
        assert pool.status()["slots"] == 3 and pool.status()["idle"] == 3
        started = time.monotonic()
        tasks = [asyncio.create_task(pool.submit(sleep_job(0.6, f"r{i}"), b"")) for i in range(4)]
        await asyncio.sleep(0.2)
        status = pool.status()
        results = await asyncio.gather(*tasks)
        return time.monotonic() - started, status, [_span(r) for r in results], pool.status()

    elapsed, during, spans, after = run_batched(scenario, slots=3)
    assert during["running"] == 3 and during["queued"] == 1 and during["idle"] == 0
    assert len({pid for pid, _, _ in spans}) == 1  # all in the one worker
    first_three = sorted(spans, key=lambda s: s[1])[:3]
    assert max(s for _, s, _ in first_three) < min(e for _, _, e in first_three)  # overlapping
    assert elapsed < 3 * 0.6  # two rounds, not four
    assert after["running"] == 0 and after["idle"] == 3


def test_requests_go_to_the_least_busy_worker():
    async def scenario(pool):
        return await asyncio.gather(*[pool.submit(sleep_job(0.3, f"r{i}"), b"") for i in range(2)])

    results = run_batched(scenario, size=2, slots=2)
    assert len({_span(r)[0] for r in results}) == 2


def test_cancelling_one_job_leaves_the_others_running():
    async def scenario(pool):
        cancel = serve.CancellationToken()
        doomed = asyncio.create_task(pool.submit(sleep_job(30), b"", cancelled=cancel.event))
        survivor = asyncio.create_task(pool.submit(sleep_job(0.8), b""))
        await asyncio.sleep(0.3)
        cancel.cancel("test cancel")
        with pytest.raises(GenerationCancelled):
            await asyncio.wait_for(doomed, 5)
        assert not survivor.done()
        await survivor
        return pool.status()

    status = run_batched(scenario, slots=2)
    assert status["idle"] == 2 and status["running"] == 0


def test_a_crash_fails_every_running_job_and_the_replacement_gets_its_slots():
    async def scenario(pool):
        running = [asyncio.create_task(pool.submit(sleep_job(30), b"")) for _ in range(2)]
        await asyncio.sleep(0.3)
        crash = asyncio.create_task(pool.submit({"action": "crash"}, b""))
        for task in running + [crash]:
            with pytest.raises(WorkerCrashed):
                await asyncio.wait_for(task, 10)
        # The replacement takes requests on all its slots again.
        results = await asyncio.wait_for(
            asyncio.gather(*[pool.submit(sleep_job(0.2), b"") for _ in range(3)]), 60)
        return results, pool.status()

    results, status = run_batched(scenario, slots=3, restart_delay=0.1)
    assert len(results) == 3
    assert status["restarts"] == 1 and status["idle"] == 3
