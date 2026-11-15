"""The staged pipeline: prepare and export on CPU workers, inference on GPU
workers, so different requests' CPU and GPU stages overlap.

Covered guarantees:
  * with one GPU worker, a request's prepare runs during another's inference,
    while inferences themselves never overlap.
  * every stage of a request gets the same seed; CPU workers see no GPU.
  * an urgent request (a request's export) waits ahead of non-urgent ones.
"""

import asyncio
import json
import time

import serve
from conftest import run_with_pool
from pool_fixtures import STAGE_SECONDS


def _spans(payload: bytes):
    upload, spans = payload.split(b"|", 1)
    return upload, [dict(zip(("stage", "request", "start", "end", "cuda"), span))
                    for span in json.loads(spans)]


def test_cpu_stages_overlap_inference_on_a_single_gpu_worker():
    requests = 3

    async def main():
        gpu = serve.WorkerPool("pool_fixtures:STAGED_GPU", 1, label="gpu-worker")
        cpu = serve.WorkerPool("pool_fixtures:STAGED_CPU", 2, cuda=False, label="cpu-worker")
        await asyncio.gather(gpu.start(), cpu.start())
        serve.state.pool, serve.state.cpu_pool = gpu, cpu
        try:
            started = time.monotonic()
            results = await asyncio.gather(*[
                serve._generate(f"upload {i}".encode(), "a.glb", serve.GenParams(), f"r{i}")
                for i in range(requests)])
            return time.monotonic() - started, results
        finally:
            await asyncio.gather(gpu.close(timeout=5), cpu.close(timeout=5))

    elapsed, results = asyncio.run(main())
    spans = []
    for i, result in enumerate(results):
        assert len(result) == 1
        upload, request_spans = _spans(result[0][0])
        assert upload == f"upload {i}".encode()  # no cross-talk between requests
        assert [s["stage"] for s in request_spans] == ["prepare", "infer", "export"]
        assert {s["request"] for s in request_spans} == {f"r{i}"}
        for stage in request_spans:
            # CPU workers see no GPU; a single-GPU pool leaves the GPU worker's as is.
            assert (stage["cuda"] == "") == (stage["stage"] != "infer")
        spans += request_spans

    infers = sorted((s for s in spans if s["stage"] == "infer"), key=lambda s: s["start"])
    for earlier, later in zip(infers, infers[1:]):
        assert later["start"] >= earlier["end"] - 0.01  # one GPU worker: one at a time
    # Some request was parsed (or exported) while another was on the GPU.
    cpu_spans = [s for s in spans if s["stage"] != "infer"]
    assert any(c["request"] != g["request"] and c["start"] < g["end"] and g["start"] < c["end"]
               for c in cpu_spans for g in infers)

    serial = requests * sum(STAGE_SECONDS.values())
    pipelined = (STAGE_SECONDS["prepare"] + requests * STAGE_SECONDS["infer"]
                 + STAGE_SECONDS["export"])
    assert elapsed < (serial + pipelined) / 2, (elapsed, serial, pipelined)


def test_urgent_requests_wait_ahead_of_others():
    order = []

    async def scenario(pool):
        async def job(tag, urgent):
            await pool.submit({"action": "echo", "request_id": tag}, b"", urgent=urgent)
            order.append(tag)

        busy = asyncio.create_task(pool.submit({"action": "sleep", "seconds": 0.5}, b""))
        await asyncio.sleep(0.1)
        waiting = []
        for tag, urgent in (("new 1", False), ("export 1", True), ("new 2", False),
                            ("export 2", True)):
            waiting.append(asyncio.create_task(job(tag, urgent)))
            await asyncio.sleep(0.02)
        await asyncio.gather(busy, *waiting)
        return pool.status()

    status = run_with_pool(scenario, size=1)
    assert order == ["export 1", "export 2", "new 1", "new 2"]
    assert status["queued"] == 0


def test_cancelled_urgent_waiter_leaves_the_queue():
    async def scenario(pool):
        busy = asyncio.create_task(pool.submit({"action": "sleep", "seconds": 0.5}, b""))
        await asyncio.sleep(0.1)
        cancel = serve.CancellationToken()
        urgent = asyncio.create_task(pool.submit({"action": "echo"}, b"", urgent=True,
                                                 cancelled=cancel.event))
        normal = asyncio.create_task(pool.submit({"action": "echo"}, b"ok"))
        await asyncio.sleep(0.05)
        cancel.cancel("client gone")
        try:
            await urgent
        except serve.GenerationCancelled:
            pass
        else:
            raise AssertionError("the cancelled request ran")
        result = await normal
        await busy
        return result, pool._urgent

    result, urgent_left = run_with_pool(scenario, size=1)
    assert result[0] == b"glb:-:ok"
    assert not urgent_left
