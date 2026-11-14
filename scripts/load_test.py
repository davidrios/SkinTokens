"""Concurrent load test for the SkinTokens WebSocket server.

Runs a closed loop: ``--concurrency`` clients each submit a request, wait for
its result, and submit the next one, until ``--duration`` seconds have passed.
Requests still running then get until ``--deadline`` (counted from the start)
to finish; any left after that are cancelled and reported as unfinished, so a
run never takes longer than the deadline (at most 120 s).

Every request uses a fixed seed (its index in the run) and the inputs rotate in
a fixed order, so two runs against different servers do the same work.

Examples:
    # One run against a server; writes its summary as JSON.
    python scripts/load_test.py --server http://localhost:8087 -c 4 --json before.json

    # Compare two summaries.
    python scripts/load_test.py --compare before.json after.json

Requires the ``websockets`` package.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HARD_LIMIT_SEC = 120.0
DEFAULT_INPUTS = [
    "examples/dragon_mesh_only.glb",
    "examples/horse_mesh_only.glb",
    "examples/buffalo_mesh_only.glb",
]


def _websocket_url(server: str) -> str:
    base = server.rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http):]
    if not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


@dataclass
class RequestResult:
    index: int
    input: str
    ok: bool
    latency_sec: float
    # Time from submit until inference started on it (the first progress
    # message at 10% or more), i.e. how long it waited for the GPU.
    queue_sec: float | None = None
    output_bytes: int = 0
    error: str | None = None


@dataclass
class Summary:
    server: str
    concurrency: int
    duration_sec: float
    elapsed_sec: float
    completed: int
    failed: int
    unfinished: int
    throughput_per_min: float
    latency: dict = field(default_factory=dict)
    queue: dict = field(default_factory=dict)
    health: dict | None = None
    errors: list = field(default_factory=list)


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]

    return {"mean": round(statistics.fmean(ordered), 2), "p50": round(pct(0.5), 2),
            "p95": round(pct(0.95), 2), "max": round(ordered[-1], 2), "min": round(ordered[0], 2)}


async def _one_request(ws_url: str, index: int, name: str, data: bytes,
                       output: str = "rig") -> RequestResult:
    import websockets

    started = time.monotonic()
    queue_sec = None
    async with websockets.connect(ws_url, max_size=None, open_timeout=30) as ws:
        try:
            await ws.send(json.dumps({"filename": Path(name).name, "seed": index,
                                      "output": output}))
            await ws.send(data)
            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                stage = event.get("stage")
                if stage == "processing" and queue_sec is None and event.get("progress", 0) >= 10:
                    queue_sec = time.monotonic() - started
                elif stage == "done":
                    result = await ws.recv()
                    size = len(result) if isinstance(result, bytes) else 0
                    if event.get("sample_index", 0) + 1 >= event.get("num_samples", 1):
                        return RequestResult(index, name, True, time.monotonic() - started,
                                             queue_sec, size)
                elif stage in ("error", "cancelled"):
                    return RequestResult(index, name, False, time.monotonic() - started,
                                         queue_sec, error=str(event.get("message")))
        except asyncio.CancelledError:
            # Deadline reached: tell the server to drop the job, then leave.
            try:
                await asyncio.wait_for(ws.send(json.dumps({"type": "cancel"})), 2)
            except Exception:
                pass
            raise


async def _health(server: str) -> dict | None:
    """GET /health without extra dependencies."""
    from urllib.request import urlopen

    url = server.rstrip("/").replace("ws://", "http://").replace("wss://", "https://") + "/health"
    try:
        return await asyncio.to_thread(lambda: json.loads(urlopen(url, timeout=5).read()))
    except Exception as exc:
        return {"error": str(exc)}


async def run(server: str, concurrency: int, duration: float, deadline: float,
              inputs: list[str], output: str = "rig") -> tuple[Summary, list[RequestResult]]:
    ws_url = _websocket_url(server)
    payloads = [(path, Path(path).read_bytes()) for path in inputs]
    counter = itertools.count()
    results: list[RequestResult] = []
    started = time.monotonic()
    stop_at = started + duration

    def log(text: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] {text}", flush=True)

    async def client(slot: int) -> None:
        while time.monotonic() < stop_at:
            index = next(counter)
            name, data = payloads[index % len(payloads)]
            try:
                result = await _one_request(ws_url, index, name, data, output)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = RequestResult(index, name, False, 0.0, error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            status = "ok " if result.ok else "ERR"
            queue = f"{result.queue_sec:5.1f}s" if result.queue_sec is not None else "   - "
            log(f"client {slot} req {index:3d} {status} latency={result.latency_sec:5.1f}s "
                f"queue={queue} {Path(name).name}" + (f" {result.error}" if result.error else ""))

    health_before = await _health(server)
    log(f"start: server={server} concurrency={concurrency} duration={duration:.0f}s "
        f"deadline={deadline:.0f}s health={health_before}")
    tasks = [asyncio.create_task(client(slot)) for slot in range(concurrency)]
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, started + deadline - time.monotonic()))
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    elapsed = time.monotonic() - started
    issued = next(counter)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    summary = Summary(
        server=server, concurrency=concurrency, duration_sec=duration,
        elapsed_sec=round(elapsed, 2), completed=len(ok), failed=len(failed),
        unfinished=issued - len(results),
        throughput_per_min=round(len(ok) / elapsed * 60, 2) if elapsed else 0.0,
        latency=_stats([r.latency_sec for r in ok]),
        queue=_stats([r.queue_sec for r in ok if r.queue_sec is not None]),
        health=health_before,
        errors=sorted({r.error for r in failed if r.error})[:10],
    )
    return summary, results


def _print_summary(summary: Summary) -> None:
    print()
    print(f"server       {summary.server}")
    print(f"concurrency  {summary.concurrency}")
    print(f"elapsed      {summary.elapsed_sec:.1f}s (submitting for {summary.duration_sec:.0f}s)")
    print(f"completed    {summary.completed}  failed {summary.failed}  unfinished {summary.unfinished}")
    print(f"throughput   {summary.throughput_per_min:.2f} req/min")
    print(f"latency      {summary.latency}")
    print(f"wait to GPU  {summary.queue}")
    if summary.errors:
        print(f"errors       {summary.errors}")


def compare(paths: list[str]) -> None:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    headers = ["run", "conc", "done", "fail", "unfin", "req/min", "lat p50", "lat p95", "wait p50"]
    table = [[Path(p).stem, r["concurrency"], r["completed"], r["failed"], r["unfinished"],
              f"{r['throughput_per_min']:.2f}", r["latency"].get("p50", "-"),
              r["latency"].get("p95", "-"), r["queue"].get("p50", "-")]
             for p, r in zip(paths, rows)]
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *table)]
    for row in [headers, *table]:
        print("  ".join(str(x).rjust(w) for x, w in zip(row, widths)))
    if len(rows) >= 2 and rows[0]["throughput_per_min"]:
        speedup = rows[-1]["throughput_per_min"] / rows[0]["throughput_per_min"]
        print(f"\nthroughput {Path(paths[-1]).stem} vs {Path(paths[0]).stem}: {speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", default="http://localhost:8087")
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds during which new requests are submitted")
    parser.add_argument("--deadline", type=float, default=110.0,
                        help=f"hard end of the run in seconds (max {HARD_LIMIT_SEC:.0f})")
    parser.add_argument("--input", action="append", dest="inputs",
                        help="input file; repeat to rotate (default: the mesh-only examples)")
    parser.add_argument("--output-format", choices=("rig", "glb"), default="rig",
                        help="result format to request (glb needs SKINTOKENS_ENABLE_GLB_OUTPUT=1)")
    parser.add_argument("--json", help="write the summary and per-request results here")
    parser.add_argument("--compare", nargs="+", metavar="JSON",
                        help="print a comparison of earlier --json summaries and exit")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    deadline = min(args.deadline, HARD_LIMIT_SEC)
    duration = min(args.duration, deadline)

    summary, results = asyncio.run(run(args.server, args.concurrency, duration, deadline,
                                       args.inputs or DEFAULT_INPUTS, args.output_format))
    _print_summary(summary)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {**asdict(summary), "requests": [asdict(r) for r in results]}, indent=2))
    sys.exit(0 if summary.failed == 0 else 1)


if __name__ == "__main__":
    main()
