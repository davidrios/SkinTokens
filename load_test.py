"""Concurrency load test for the SkinTokens WebSocket server (client-only).

At every concurrency level, sends a number of requests drawn at random from the
input files, from N concurrent clients. For each request it records the latency,
the time spent waiting for a worker, and a hash of the output.

Every request uses a fixed seed (``--seed`` plus its index in the run), so the
same input and seed should produce identical bytes at any concurrency on one
build. ``compare`` reports any (input, seed) pair whose hashes differ across the
reports it is given.

Usage::

    python load_test.py                          # defaults: --concurrency 1,4 --requests 12,24
    python load_test.py --concurrency 1,2,4 --requests 8
    python load_test.py --label baseline --out baseline.json
    python load_test.py compare baseline.json after.json

``--requests`` takes one count per concurrency level; a single value applies to
every level. The draw is seeded (``--draw-seed``), so two runs with the same
arguments send the same requests. A single warmup request runs first
unless ``--no-warmup``.

Requires the ``websockets`` package.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from urllib.request import urlopen

DEFAULT_INPUTS = [
    "examples/dragon_mesh_only.glb",
    "examples/horse_mesh_only.glb",
    "examples/buffalo_mesh_only.glb",
]
# Inference starts at 10% progress; everything before is queueing and parsing.
INFERENCE_START_PERCENT = 10


def websocket_url(server: str) -> str:
    base = server.rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http):]
    if not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


def _fetch_health(server: str) -> dict | None:
    url = server.rstrip("/").replace("ws://", "http://").replace("wss://", "https://") + "/health"
    try:
        return json.loads(urlopen(url, timeout=5).read())
    except Exception as exc:
        return {"error": str(exc)}


async def fetch_health(server: str) -> dict | None:
    # urlopen blocks; run it off the event loop so it cannot stall the requests.
    return await asyncio.to_thread(_fetch_health, server)


def load_inputs(paths: list[str]) -> dict[str, bytes]:
    inputs = {p: Path(p).read_bytes() for p in paths}
    if not inputs:
        raise SystemExit("no input files")
    return inputs


async def one_request(server: str, name: str, data: bytes, seed: int, output: str,
                      origin: float) -> dict:
    import websockets

    row: dict = {"input": name, "seed": seed, "ok": False}
    started = time.perf_counter()
    row["t_start"] = round(started - origin, 3)
    hasher = hashlib.sha256()
    try:
        async with websockets.connect(websocket_url(server),
                                      max_size=None, open_timeout=30,
                                      compression=None) as ws:
            await ws.send(json.dumps({"filename": Path(name).name, "seed": seed,
                                      "output": output}))
            await ws.send(data)
            expected, received, size = 1, 0, 0
            async for raw in ws:
                now = time.perf_counter() - started
                if isinstance(raw, bytes):
                    hasher.update(raw)
                    size += len(raw)
                    received += 1
                    if received >= expected:
                        row["ok"] = True
                        break
                    continue
                message = json.loads(raw)
                row.setdefault("request_id", message.get("request_id"))
                stage = message.get("stage")
                if stage == "processing":
                    if "wait_sec" not in row and message.get("progress", 0) >= INFERENCE_START_PERCENT:
                        row["wait_sec"] = round(now, 3)
                elif stage == "done":
                    expected = int(message.get("num_samples", 1))
                elif stage in ("error", "cancelled"):
                    row["error"] = message.get("message") or f"server sent {stage}"
                    break
            row["output_sha256"] = hasher.hexdigest()
            row["output_size"] = size
    except Exception as exc:  # A failed request is recorded, not fatal.
        row["error"] = f"{type(exc).__name__}: {exc}"
    if not row["ok"]:
        row.setdefault("error", "connection closed before a result")
    ended = time.perf_counter()
    row["t_end"] = round(ended - origin, 3)
    row["latency"] = round(ended - started, 3)
    if row["ok"] and "wait_sec" in row:
        row["gpu_sec"] = round(row["latency"] - row["wait_sec"], 3)
    return row


async def run_level(server: str, inputs: dict[str, bytes], concurrency: int, count: int,
                    rng: random.Random, base_seed: int, output: str) -> list[dict]:
    names = list(inputs)
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for index in range(count):
        queue.put_nowait((rng.choice(names), base_seed + index))
    rows: list[dict] = []
    origin = time.perf_counter()

    async def worker() -> None:
        while True:
            try:
                name, seed = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            rows.append(await one_request(server, name, inputs[name], seed, output, origin))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    wall = max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)
    latencies = [r["latency"] for r in ok]
    waits = [r["wait_sec"] for r in ok if "wait_sec" in r]
    summary = {
        "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "wall_sec": round(wall, 3),
        "throughput_rps": round(len(ok) / wall, 3) if wall > 0 else 0.0,
        "latency_p50": round(percentile(latencies, .5), 3),
        "latency_p95": round(percentile(latencies, .95), 3),
        "latency_max": round(max(latencies, default=float("nan")), 3),
        "wait_mean": round(statistics.fmean(waits), 3) if waits else None,
        "by_input": {},
    }
    for name in sorted({r["input"] for r in ok}):
        group = [r for r in ok if r["input"] == name]
        entry = {"n": len(group),
                 "latency_p50": round(percentile([r["latency"] for r in group], .5), 3)}
        if any("gpu_sec" in r for r in group):
            entry["gpu_sec"] = round(statistics.fmean(r["gpu_sec"] for r in group if "gpu_sec" in r), 3)
        summary["by_input"][name] = entry
    errors = sorted({str(r.get("error")) for r in rows if not r["ok"]})
    if errors:
        summary["error_messages"] = errors[:5]
    return summary


def level_counts(concurrency: str, requests: str) -> list[tuple[int, int]]:
    """``(concurrency, requests)`` per level; one request count applies to all."""
    levels = [int(c) for c in concurrency.split(",")]
    counts = [int(n) for n in requests.split(",")]
    if len(counts) == 1:
        counts *= len(levels)
    if len(counts) != len(levels):
        raise SystemExit(f"--requests has {len(counts)} values for {len(levels)} concurrency levels")
    if min(counts) < 1 or min(levels) < 1:
        raise SystemExit("concurrency levels and request counts must be at least 1")
    return list(zip(levels, counts))


async def cmd_run(args) -> int:
    inputs = load_inputs(args.inputs or DEFAULT_INPUTS)
    levels = level_counts(args.concurrency, args.requests)
    rng = random.Random(args.draw_seed)
    out_path = Path(args.out) if args.out else None
    health = await fetch_health(args.server)
    print(f"[load] server={args.server} health={health}", flush=True)
    report = {"label": args.label, "server": args.server, "output": args.output_format,
              "health": health, "runs": []}
    prefix = f"{args.label} " if args.label else ""
    failed = False
    if args.warmup:
        name, data = next(iter(inputs.items()))
        row = await one_request(args.server, name, data, args.seed, args.output_format,
                                time.perf_counter())
        status = "ok" if row["ok"] else f"ERROR {row.get('error')}"
        failed |= not row["ok"]
        print(f"[load] warmup {name}: {row['latency']:.2f}s {status}", flush=True)
    for concurrency, count in levels:
        rows = await run_level(args.server, inputs, concurrency, count, rng, args.seed,
                               args.output_format)
        summary = summarize(rows)
        failed |= summary["errors"] > 0
        report["runs"].append({"concurrency": concurrency, "summary": summary, "requests": rows})
        print(f"[load] {prefix}c={concurrency}: "
              f"{summary['ok']}/{summary['requests']} ok, "
              f"{summary['throughput_rps']:.3f} req/s, p50 {summary['latency_p50']:.2f}s, "
              f"p95 {summary['latency_p95']:.2f}s, wait {summary['wait_mean']}s", flush=True)
        for message in summary.get("error_messages", []):
            print(f"[load]   error: {message}", flush=True)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_path is not None:
        print(f"[load] saved {out_path}")
    return 1 if failed else 0


def cmd_compare(args) -> None:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    labels = [r.get("label") or Path(p).stem for r, p in zip(reports, args.reports)]
    levels = sorted({run["concurrency"] for r in reports for run in r["runs"]})
    metrics = [("throughput_rps", "req/s"), ("latency_p50", "p50 s"),
               ("latency_p95", "p95 s"), ("wait_mean", "wait s")]
    width = max(10, *(len(label) for label in labels))
    print(f"{'c':>3} {'metric':<7} " + " ".join(f"{label:>{width}}" for label in labels)
          + f" {'last/first':>10}")
    for concurrency in levels:
        for metric, title in metrics:
            values = []
            for report in reports:
                run = next((x for x in report["runs"] if x["concurrency"] == concurrency), None)
                values.append(None if run is None else run["summary"].get(metric))
            cells = " ".join(f"{'-' if v is None else v:>{width}}" for v in values)
            ratio = f"{values[-1] / values[0]:.2f}x" if values[0] and values[-1] is not None else ""
            print(f"{concurrency:>3} {title:<7} {cells} {ratio:>10}")
    # The same input and seed must produce the same bytes everywhere on one build.
    hashes: dict[tuple, set] = {}
    for report in reports:
        for run in report["runs"]:
            for row in run["requests"]:
                if row["ok"]:
                    hashes.setdefault((row["input"], row["seed"]), set()).add(row["output_sha256"])
    differing = {key: len(found) for key, found in hashes.items() if len(found) > 1}
    print("\n## output bytes per (input, seed)")
    if differing:
        for (name, seed), n in sorted(differing.items()):
            print(f"  {name} seed={seed}: {n} distinct outputs")
    else:
        print(f"  all {len(hashes)} (input, seed) pairs produced identical bytes in every run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Send random requests at several concurrency levels")
    run.add_argument("--server", default="http://localhost:8087")
    run.add_argument("--input", action="append", dest="inputs",
                     help="Input file; repeat for several (default: the mesh-only examples)")
    run.add_argument("--concurrency", default="1,4", help="Comma-separated concurrency levels")
    run.add_argument("--requests", default="12,24",
                     help="Requests per level, comma-separated like --concurrency "
                          "(one value applies to all levels)")
    run.add_argument("--seed", type=int, default=0,
                     help="Base seed; request i of a level uses seed + i")
    run.add_argument("--draw-seed", type=int, default=0, help="Seed of the random input draw")
    run.add_argument("--output-format", choices=("rig", "glb"), default="rig",
                     help="Result format to request (glb needs SKINTOKENS_ENABLE_GLB_OUTPUT=1)")
    run.add_argument("--label", help="Name stored in the report and shown in progress lines")
    run.add_argument("--out", help="Save the JSON report here (console only if omitted)")
    run.add_argument("--no-warmup", dest="warmup", action="store_false")
    compare = sub.add_parser("compare", help="Compare saved reports side by side")
    compare.add_argument("reports", nargs="+")
    argv = sys.argv[1:]
    # "run" is the default: any argv not starting with a subcommand is run's arguments.
    if not argv or argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(cmd_run(args))
    cmd_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
