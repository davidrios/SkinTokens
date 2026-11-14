"""
SkinTokens / TokenRig inference server — FastAPI persistent service.

This process is the gateway: it holds the WebSocket connections and loads no
model. Generation runs in a pool of worker processes (``src/server/pool.py``),
each with its own copy of the model resident in VRAM and running one request at
a time, so up to ``SKINTOKENS_WORKERS`` requests are parsed, sampled on the GPU
and exported fully in parallel. Further requests wait for a free worker in
FIFO order and get a once-per-second queue heartbeat meanwhile.

Clients submit a 3D file (GLB/GLTF/OBJ) and receive a rigged GLB over WebSocket.
See ``src/server/pipeline.py`` for the model side (and a note on why results
are non-deterministic).

Run:
    python serve.py --host 0.0.0.0 --port 8087

Environment variables:
    SKINTOKENS_WORKERS        Worker processes / parallel requests (default 2)
    SKINTOKENS_MODEL_CKPT     Path to the model checkpoint
    SKINTOKENS_HF_PATH        Optional HuggingFace model path
    SKINTOKENS_DEVICE         Device override (default: cuda if available)
"""

import argparse
import os
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger("skintokens.serve")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.server import config
from src.server.config import GenParams, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, SUPPORTED_EXT
from src.server.pool import WorkerPool
from src.server.worker import GenerationCancelled

WORKER_SPEC = "src.server.pipeline:SERVICE"


# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
class ServerState:
    pool: Optional[WorkerPool] = None
    ready: bool = False
    loaded_at: float = 0.0
    active_jobs: int = 0

    @property
    def busy(self) -> bool:
        return self.active_jobs > 0


state = ServerState()


class CancellationToken:
    """A request's cancellation flag, on the event loop."""

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason = "generation cancelled"

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    def cancel(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            self.event.set()

    async def wait(self) -> None:
        await self.event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self.reason)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _validate_upload(file_data: bytes, filename: str) -> str:
    """The upload's (supported) suffix; the client's filename is used for nothing else."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_EXT}")
    if len(file_data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded file is too large: {len(file_data)} bytes "
                         f"(max {MAX_UPLOAD_MB} MB)")
    return suffix


async def _generate(
    file_data: bytes,
    filename: str,
    params: GenParams,
    request_id: str,
    progress_callback=None,
    cancellation: Optional[CancellationToken] = None,
    queue_callback=None,
):
    """Run one generation on the next free worker.

    ``progress_callback(percent, stage, elapsed)`` is called on the event loop
    for each progress event of the job. ``queue_callback`` (async) is invoked
    once per second with ``(phase, waited_sec, queue_ahead)`` while the request
    waits for a worker.

    Returns:
        List of (glb_bytes, meta_dict) tuples, one per sample (meta is empty).
    """
    cancellation = cancellation or CancellationToken()
    suffix = _validate_upload(file_data, filename)
    cancellation.raise_if_cancelled()
    if state.pool is None:
        raise RuntimeError("model still loading")

    logger.info("[%s] submitted queued=%s filename=%r", request_id,
                state.pool.would_queue(), filename)
    wait_started = time.monotonic()

    async def on_start():
        logger.info("[%s] worker acquired after %.2fs", request_id,
                    time.monotonic() - wait_started)

    def on_progress(event: dict):
        if progress_callback is not None:
            progress_callback(event["progress"], event["step"], event["elapsed_sec"])

    state.active_jobs += 1
    try:
        glbs = await state.pool.submit(
            {**params.model_dump(), "suffix": suffix, "request_id": request_id},
            file_data,
            on_start=on_start,
            on_progress=on_progress,
            queue_callback=queue_callback,
            queue_phase="worker",
            cancelled=cancellation.event,
            cancel_reason=lambda: cancellation.reason,
        )
    except GenerationCancelled:
        raise GenerationCancelled(cancellation.reason) from None
    finally:
        state.active_jobs -= 1
    return [(glb, {}) for glb in glbs]


async def _watch_ws_cancellation(ws: WebSocket, cancellation: CancellationToken,
                                 request_id: str) -> bool:
    """Return True for an explicit cancel message, False for a disconnect."""
    try:
        while True:
            message = await ws.receive_json()
            message_type = message.get("type", message.get("action", ""))
            if str(message_type).lower() in {"cancel", "interrupt"}:
                logger.info("[%s] explicit cancellation requested", request_id)
                cancellation.cancel("client requested cancellation")
                return True
            logger.warning("[%s] ignoring WebSocket message type=%r", request_id, message_type)
    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected; cancelling generation", request_id)
        cancellation.cancel("WebSocket client disconnected")
        return False
    except RuntimeError:
        cancellation.cancel("WebSocket client disconnected")
        return False


class _WsOutbox:
    """Sends a connection's JSON messages in order from one task, so progress
    reported by callbacks never blocks (or reorders around) anything else."""

    def __init__(self, ws: WebSocket, cancellation: CancellationToken) -> None:
        self.ws = ws
        self.cancellation = cancellation
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task = asyncio.create_task(self._pump())

    def post(self, message: dict) -> None:
        self.queue.put_nowait(message)

    async def _pump(self) -> None:
        while True:
            message = await self.queue.get()
            try:
                await self.ws.send_json(message)
            except Exception:
                self.cancellation.cancel("WebSocket client disconnected")
            finally:
                self.queue.task_done()

    async def flush(self) -> None:
        await self.queue.join()

    async def close(self) -> None:
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def _worker_options() -> dict:
    return {"model_ckpt": config.MODEL_CKPT, "hf_path": config.HF_PATH,
            "cpu_threads": config.worker_cpu_threads(config.WORKERS)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %d worker process%s", config.WORKERS,
                "" if config.WORKERS == 1 else "es")
    pool = WorkerPool(WORKER_SPEC, config.WORKERS, options=_worker_options())
    await pool.start()
    state.pool = pool
    state.ready = True
    state.loaded_at = time.time()
    logger.info("Pipeline ready: %d worker%s (%s)", config.WORKERS,
                "" if config.WORKERS == 1 else "s", pool.info)
    try:
        yield
    finally:
        state.ready = False
        state.pool = None
        await pool.close()


app = FastAPI(title="SkinTokens / TokenRig Inference Server", version="1.0",
              lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready or state.pool is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    pool = state.pool.status()
    return {
        "status": "ok",
        "busy": state.busy,
        "active_jobs": state.active_jobs,
        "workers": pool["workers"],
        "pool": pool,
        "queue_depth": pool["queued"],
    }


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    WebSocket protocol:
      client -> {"filename": "model.obj", ...params}
      client -> <binary: file data>
      client -> {"type": "cancel"}
      server -> {"stage": "queued"|"processing"|"done"|"cancelled"|"error", ...}
      server -> {"stage": "done", "glb_size": N, ...}
      server -> <binary: GLB data>
    """
    request_id = uuid.uuid4().hex[:8]
    cancellation = None
    generation_task = None
    receiver_task = None
    outbox = None

    await ws.accept()
    logger.info("[%s] WebSocket connected client=%s", request_id, ws.client)

    try:
        req = await ws.receive_json()
        logger.info("[%s] WebSocket generation request received", request_id)

        if not state.ready or state.pool is None:
            logger.warning("[%s] rejected: model still loading", request_id)
            await ws.send_json({"stage": "error", "message": "model still loading"})
            await ws.close()
            return

        filename = req.pop("filename", "input.obj")
        if req.get("use_skeleton"):
            logger.warning("[%s] rejected: use_skeleton is no longer supported", request_id)
            await ws.send_json({"stage": "error", "message":
                                "use_skeleton is no longer supported; inputs are meshes and "
                                "the skeleton is always generated"})
            await ws.close()
            return

        try:
            file_data = await ws.receive_bytes()
        except Exception as e:
            logger.warning("[%s] invalid file: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid file: {e}"})
            await ws.close()
            return

        params = GenParams(**{k: v for k, v in req.items() if k in GenParams.model_fields})
        logger.info("[%s] file decoded bytes=%d filename=%s params=%s",
                    request_id, len(file_data), filename, params.model_dump())

        cancellation = CancellationToken()
        outbox = _WsOutbox(ws, cancellation)

        def send_progress(percent, stage, elapsed):
            outbox.post({"stage": "processing", "step": stage, "progress": percent,
                         "elapsed_sec": elapsed, "request_id": request_id})

        async def send_queue(phase, waited_sec, queue_ahead):
            outbox.post({"stage": "queued", "phase": phase, "queued": True,
                         "waited_sec": waited_sec, "queue_ahead": queue_ahead,
                         "request_id": request_id})

        queued = state.pool.would_queue()
        logger.info("[%s] queued=%s", request_id, queued)
        outbox.post({"stage": "queued", "queued": queued, "request_id": request_id})
        outbox.post({"stage": "processing", "progress": 0, "request_id": request_id})

        t0 = time.time()
        generation_task = asyncio.create_task(_generate(
            file_data, filename, params, request_id, send_progress,
            cancellation=cancellation, queue_callback=send_queue,
        ))
        del file_data
        receiver_task = asyncio.create_task(
            _watch_ws_cancellation(ws, cancellation, request_id))
        done, _ = await asyncio.wait((generation_task, receiver_task),
                                     return_when=asyncio.FIRST_COMPLETED)

        explicit_cancel = False
        if receiver_task in done:
            explicit_cancel = receiver_task.result()

        if cancellation.cancelled:
            try:
                await generation_task
            except GenerationCancelled:
                pass
            if explicit_cancel:
                await outbox.flush()
                await ws.send_json({"stage": "cancelled", "message": cancellation.reason,
                                    "request_id": request_id})
                await ws.close()
            logger.info("[%s] WebSocket generation cancelled: %s", request_id, cancellation.reason)
            return

        receiver_task.cancel()
        with suppress(asyncio.CancelledError):
            await receiver_task

        results = await generation_task  # list of (glb, meta) tuples
        await outbox.flush()

        for sample_idx, (glb, _meta) in enumerate(results):
            await ws.send_json({
                "stage": "done",
                "sample_index": sample_idx,
                "num_samples": len(results),
                "elapsed_sec": round(time.time() - t0, 2),
                "progress": 100,
                "request_id": request_id,
                "glb_size": len(glb),
            })
            await ws.send_bytes(glb)

        logger.info("[%s] WebSocket request completed samples=%d elapsed=%.2fs",
                    request_id, len(results), time.time() - t0)
        await ws.close()

    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected", request_id)
    except GenerationCancelled as e:
        logger.info("[%s] WebSocket generation cancelled: %s", request_id, e)
    except Exception as e:
        logger.exception("[%s] WebSocket generation failed", request_id)
        try:
            if outbox is not None:
                await outbox.flush()
            await ws.send_json({"stage": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass
    finally:
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task
        if generation_task is not None and not generation_task.done():
            if cancellation is not None:
                cancellation.cancel("WebSocket handler stopped")
            with suppress(asyncio.CancelledError, GenerationCancelled, Exception):
                await asyncio.shield(generation_task)
        if outbox is not None:
            await outbox.close()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SkinTokens / TokenRig inference server")
    parser.add_argument("--host", default=os.environ.get("SKINTOKENS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SKINTOKENS_PORT", "8087")))
    parser.add_argument("--model-ckpt", default=config.MODEL_CKPT)
    parser.add_argument("--hf-path", default=config.HF_PATH)
    parser.add_argument("--workers", type=int, default=config.WORKERS,
                        help="worker processes (parallel requests); SKINTOKENS_WORKERS")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config.MODEL_CKPT = args.model_ckpt
    config.HF_PATH = args.hf_path
    config.WORKERS = args.workers

    import uvicorn

    class HealthCheckFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return '"GET /health' not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"] = {
        name: {**formatter, "fmt": f"%(asctime)s.%(msecs)03d {formatter['fmt']}",
               "datefmt": LOG_DATE_FORMAT}
        for name, formatter in uvicorn.config.LOGGING_CONFIG["formatters"].items()
    }
    # One gateway process (uvicorn workers=1): the model workers are the pool's.
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=MAX_UPLOAD_BYTES + 64 * 1024, log_config=log_config)


if __name__ == "__main__":
    main()
