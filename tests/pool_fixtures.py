"""Worker specs for tests that start real worker processes.

They load no model, so a worker starts in about a second. Imported by spawned
processes, so keep it free of test-only imports.
"""
import os
import time

from src.server.worker import ServiceSpec


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
