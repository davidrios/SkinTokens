"""Worker specs for tests that start real worker processes.

They load no model, so a worker starts in about a second. Imported by spawned
processes, so keep it free of test-only imports.
"""
import os
import time

from src.server.worker import ServiceSpec, main as worker_main


def close_pipe_before_finalizers(*args):
    """Simulate IPC closing before model/compiler process-exit hooks finish."""
    try:
        worker_main(*args)
    finally:
        args[1].close()
        args[2].close()


def _load(options):
    if options.get("fail"):
        raise RuntimeError("cannot load the fixture model")
    return {"fixture": True}


def _run(params, upload, cancellation, progress):
    action = params.get("action", "echo")
    if action == "echo":
        progress({"progress": 50, "step": "half way", "elapsed_sec": 0.0})
        return [b"glb:" + params.get("request_id", "-").encode() + b":" + upload,
                str(os.getpid()).encode()]
    if action == "sleep":
        # Sleeps in slices, checking for cancellation like the real pipeline
        # does at each progress report.
        started = time.time()
        deadline = time.monotonic() + params["seconds"]
        while time.monotonic() < deadline:
            cancellation.raise_if_cancelled()
            time.sleep(0.01)
        return [str(os.getpid()).encode(), repr((started, time.time())).encode()]
    if action == "crash":
        os._exit(3)
    if action == "raise":
        raise ValueError("bad request value")
    raise AssertionError(f"unknown action {action}")


ECHO = ServiceSpec("echo", _load, _run)


def _load_resources(options):
    import multiprocessing
    from multiprocessing.util import Finalize

    # Keep named semaphores alive until normal process teardown. A worker
    # killed by group-wide SIGTERM cannot run their unlink finalizers.
    global _resources
    ctx = multiprocessing.get_context("spawn")
    _resources = [ctx.Semaphore(1), ctx.Semaphore(1)]
    if options.get("finalize_delay"):
        Finalize(None, _finalize_resources,
                 args=(options["finalize_delay"], options["finalize_marker"],
                       options.get("ignore_term", False)), exitpriority=10)
    return _load(options)


def _finalize_resources(delay, marker, ignore_term):
    import signal
    from pathlib import Path

    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(marker).write_text("started")
    time.sleep(delay)
    Path(marker).write_text("complete")


RESOURCES = ServiceSpec("resources", _load_resources, _run)


def _idle():
    """Housekeeping that runs after the reply: slow, then leaves a marker."""
    time.sleep(0.5)
    with open(os.environ["IDLE_MARKER"], "w") as f:
        f.write("idle")


IDLE = ServiceSpec("idle", _load, _run, _idle)


# The staged pipeline (serve._generate) with stand-in stages: each sleeps for
# the seconds given in the stage durations below and hands on the spans
# (stage, request, start, end, CUDA_VISIBLE_DEVICES) every stage so far ran in.
STAGE_SECONDS = {"prepare": 0.4, "infer": 0.6, "export": 0.4}


def _stage_span(stage, params, cancellation):
    started = time.time()
    deadline = time.monotonic() + STAGE_SECONDS[stage]
    while time.monotonic() < deadline:
        cancellation.raise_if_cancelled()
        time.sleep(0.01)
    return [stage, params["request_id"], started, time.time(),
            os.environ.get("CUDA_VISIBLE_DEVICES")]


def _run_cpu_stage(params, payload, cancellation, progress):
    import json
    stage = params["stage"]
    if stage == "prepare":
        assert params["seed"] is not None and params["suffix"] == ".glb"
        span = _stage_span("prepare", params, cancellation)
        progress({"progress": 5, "step": "building dataset", "elapsed_sec": 0.0})
        return {"infer": json.dumps([span]).encode(), "export": payload}
    context, spans = payload
    spans = json.loads(spans) + [_stage_span("export", params, cancellation)]
    return [context + b"|" + json.dumps(spans).encode()]


def _run_gpu_stage(params, payload, cancellation, progress):
    import json
    assert "stage" not in params and params["seed"] is not None
    return json.dumps(json.loads(payload) + [_stage_span("infer", params, cancellation)]).encode()


STAGED_CPU = ServiceSpec("staged-cpu", _load, _run_cpu_stage)
STAGED_GPU = ServiceSpec("staged-gpu", _load, _run_gpu_stage)
