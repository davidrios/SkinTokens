"""
SkinTokens / TokenRig inference server — FastAPI persistent service.

This process is the gateway: it holds the WebSocket connections and loads no
model. A request runs in three stages on two pools of worker processes
(``src/server/pool.py``), so the CPU and GPU work of different requests
overlap:

1. ``prepare`` on a CPU worker: parse the upload, sample the point clouds.
2. ``infer`` on a GPU worker: sample the rig, interpolate its skin.
3. ``export`` on a CPU worker: build the rig payload or GLB.

Each GPU worker has its own copy of the model resident in VRAM and runs up
to ``SKINTOKENS_BATCH_SLOTS`` (1, 2, 3 or 4) requests' inference at once, decoding
their point clouds together with continuous batching (``src/server/pipeline.py``);
the GPU pool runs ``SKINTOKENS_WORKERS`` workers **on each** visible GPU
(pinned round-robin, each worker sees its card as ``cuda:0``). The CPU workers (``SKINTOKENS_CPU_WORKERS``) load no model and
see no GPU (``src/server/stages.py``). So with one GPU worker, the next
request is parsed while the current one is on the GPU, and the previous one
is exported meanwhile. A request waits for each stage's free worker in FIFO
order (exports ahead of new parses) and gets a once-per-second queue
heartbeat meanwhile.

Clients submit a 3D file (GLB/GLTF) and receive, over WebSocket, a rig payload
(skeleton and skin only; the client adds it to the file it uploaded, see
``src/rig_package/glb_io/rig_payload.py``) or, if enabled, the whole rigged
GLB. See ``src/server/pipeline.py`` for the model side (and a note on how far
results are reproducible).

Run:
    python serve.py --host 0.0.0.0 --port 8087

Environment variables:
    SKINTOKENS_WORKERS        GPU worker processes per GPU (default 1); total
                              GPU workers = this times the number of visible GPUs
    SKINTOKENS_BATCH_SLOTS    Requests (point clouds) each GPU worker decodes at
                              once: 1, 2, 3 or 4 (default 4), 0 for no batching;
                              needs fast decode on CUDA
    SKINTOKENS_CPU_WORKERS    CPU worker processes for parsing and export
                              (default 0 = twice the GPU workers)
    SKINTOKENS_MODEL_CKPT     Path to the model checkpoint
    SKINTOKENS_HF_PATH        Optional HuggingFace model path
    SKINTOKENS_DEVICE         Device override (default: cuda if available)
    SKINTOKENS_ENABLE_GLB_OUTPUT  1 to also serve "output": "glb" (whole GLBs,
                              several times larger); default: rig payloads only
    SKINTOKENS_MAX_QUEUE      Requests that may wait beyond the ones the workers
                              hold (default 16); more are refused with
                              "server busy"
    SKINTOKENS_HEADER_TIMEOUT_SEC / SKINTOKENS_UPLOAD_TIMEOUT_SEC
                              Time a client has to send its header (30) / file (300)
    SKINTOKENS_MAX_UPLOAD_MB, SKINTOKENS_INPUT_*  Upload size and geometry limits
                              (src/rig_package/glb_io/limits.py)

The server has no authentication and accepts WebSocket connections from any
origin: expose it only on a trusted network or behind an authenticating proxy.
"""

import argparse
import json
import os
import asyncio
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Optional

from src.server.logs import (LOG_DATE_FORMAT, RequestTrace, bind_trace, configure_logging,
                             save_request_dump, unbind_trace)

configure_logging()
logger = logging.getLogger("skintokens.serve")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.server import config
from src.server.config import GenParams, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, SUPPORTED_EXT
from src.server.pool import WorkerPool
from src.server.worker import GenerationCancelled

WORKER_SPEC = "src.server.pipeline:SERVICE"  # the GPU stage
CPU_WORKER_SPEC = "src.server.stages:SERVICE"  # the CPU stages


def new_request_id() -> str:
    """A per-request id for logs, failure dumps and client responses.

    The local ``HHMMSS`` prefix (same clock as the dump date directories) makes a
    request's time of day visible at a glance, and the 64-bit random suffix
    (16 hex chars) keeps the birthday-collision odds negligible even over
    millions of requests, so two concurrent requests never share a dump dir."""
    return f"{time.strftime('%H%M%S', time.localtime())}-{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
class ServerState:
    pool: Optional[WorkerPool] = None  # GPU workers: inference
    cpu_pool: Optional[WorkerPool] = None  # CPU workers: parsing and export
    ready: bool = False
    loaded_at: float = 0.0
    active_jobs: int = 0
    # Requests past their header, holding (or receiving) an upload: running,
    # waiting for a worker or still uploading. Capped by ``_admission_limit``.
    admitted: int = 0

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
def _parse_text_message(text) -> dict:
    """A client's JSON text message as a dict, bounded in size before parsing
    (the gateway's event loop parses it, so it must stay cheap)."""
    if not isinstance(text, str):
        raise ValueError("expected a JSON text message")
    if len(text) > config.MAX_TEXT_MESSAGE_BYTES:
        raise ValueError(f"JSON message too large (max {config.MAX_TEXT_MESSAGE_BYTES} bytes)")
    try:
        message = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError("invalid JSON message") from None
    if not isinstance(message, dict):
        raise ValueError("expected a JSON object")
    return message


def _request_filename(req: dict) -> str:
    filename = req.pop("filename", "input.glb")
    if not isinstance(filename, str) or len(filename) > config.MAX_FILENAME_CHARS:
        raise ValueError(f"filename must be a string of at most {config.MAX_FILENAME_CHARS} characters")
    return filename


def _unsupported_output_message(output: str) -> str:
    supported = ", ".join(f'"{f}"' for f in config.OUTPUT_FORMATS)
    message = f'output "{output}" is not supported by this server; supported: {supported}'
    if output == "glb":
        message += ' (send "output": "rig" and add the rig to the upload on the client)'
    return message


def visible_gpu_count() -> int:
    """The number of GPUs the pool spreads across.

    ``torch.cuda.device_count()`` honours ``CUDA_VISIBLE_DEVICES`` and never
    creates a CUDA context, so the gateway (which loads no model) can call it.
    Returns 1 when no CUDA is visible, so a single-GPU / CPU host still starts
    one worker per GPU.
    """
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return max(1, int(count))


def _gpu_workers() -> int:
    return config.WORKERS_PER_GPU * visible_gpu_count()


def _gpu_slots() -> int:
    """Requests the GPU workers run at once: their reported slots once the
    pool is up, else what the configuration asks for."""
    status = getattr(state.pool, "status", None)
    if status is not None:
        slots = status().get("slots")
        if slots:
            return slots
    return _gpu_workers() * max(1, config.BATCH_SLOTS)


def _admission_limit() -> int:
    gpu_workers = _gpu_workers()
    return _gpu_slots() + config.cpu_workers(gpu_workers) + config.MAX_QUEUE


def _client_error_message(exc: BaseException, request_id: str) -> str:
    """What a client is told about a failure: the message of an invalid request
    or input (``ValueError``, including parameter validation), else only the
    error type; details of internal errors stay in the server log."""
    if isinstance(exc, ValueError):
        return str(exc)
    return f"internal error ({type(exc).__name__}); request {request_id}"


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
    """Run one generation: ``prepare`` on a CPU worker, ``infer`` on a GPU
    worker, ``export`` on a CPU worker, each on the next free one.

    ``progress_callback(percent, stage, elapsed)`` is called on the event loop
    for each progress event of the stages, ``elapsed`` counted from when the
    first stage started. ``queue_callback`` (async) is invoked once per second
    with ``(phase, waited_sec, queue_ahead)`` while the request waits for a
    worker, ``phase`` being the stage it waits for.

    The stages hand their data on as packed bytes the gateway only forwards.

    Returns:
        List of (payload_bytes, meta_dict) tuples, one per sample (meta is
        empty): rig payloads or GLBs, as ``params.output`` asks.
    """
    cancellation = cancellation or CancellationToken()
    suffix = _validate_upload(file_data, filename)
    cancellation.raise_if_cancelled()
    if state.pool is None or state.cpu_pool is None:
        raise RuntimeError("model still loading")

    # Every stage samples under seeds derived from this one.
    job = {**params.with_seed().model_dump(), "request_id": request_id}
    logger.debug("[%s] submitted queued=%s filename=%r", request_id,
                 state.cpu_pool.would_queue(), filename)
    wait_started = time.monotonic()
    started: Optional[float] = None

    async def on_start():
        nonlocal started
        started = time.monotonic()
        logger.debug("[%s] CPU worker acquired after %.2fs", request_id,
                     started - wait_started)

    def on_progress(event: dict):
        if progress_callback is not None:
            progress_callback(event["progress"], event["step"],
                              round(time.monotonic() - (started or wait_started), 2))

    stage_options = dict(
        on_progress=on_progress,
        queue_callback=queue_callback,
        cancelled=cancellation.event,
        cancel_reason=lambda: cancellation.reason,
    )
    state.active_jobs += 1
    try:
        prepared = await state.cpu_pool.submit(
            {**job, "stage": "prepare", "suffix": suffix}, file_data,
            on_start=on_start, queue_phase="prepare", **stage_options)
        del file_data
        predictions = await state.pool.submit(
            job, prepared["infer"], queue_phase="infer", **stage_options)
        # A request near its end goes ahead of new requests' parsing.
        outputs = await state.cpu_pool.submit(
            {**job, "stage": "export"}, (prepared["export"], predictions),
            queue_phase="export", urgent=True, **stage_options)
    except GenerationCancelled:
        raise GenerationCancelled(cancellation.reason) from None
    finally:
        state.active_jobs -= 1
    return [(output, {}) for output in outputs]


async def _watch_ws_cancellation(ws: WebSocket, cancellation: CancellationToken,
                                 request_id: str) -> bool:
    """Return True for an explicit cancel message, False for a disconnect."""
    try:
        while True:
            try:
                message = _parse_text_message(await ws.receive_text())
            except (ValueError, KeyError):  # KeyError: a binary frame
                logger.debug("[%s] ignoring invalid WebSocket message", request_id)
                continue
            message_type = message.get("type", message.get("action", ""))
            if str(message_type).lower() in {"cancel", "interrupt"}:
                logger.debug("[%s] explicit cancellation requested", request_id)
                cancellation.cancel("client requested cancellation")
                return True
            logger.debug("[%s] ignoring WebSocket message type=%r", request_id, message_type)
    except WebSocketDisconnect:
        logger.debug("[%s] WebSocket disconnected; cancelling generation", request_id)
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
def _cpu_threads() -> int:
    gpu_workers = _gpu_workers()
    return config.worker_cpu_threads(gpu_workers + config.cpu_workers(gpu_workers))


def _worker_options() -> dict:
    return {"model_ckpt": config.MODEL_CKPT, "hf_path": config.HF_PATH,
            "cpu_threads": _cpu_threads(), "batch_slots": config.BATCH_SLOTS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpus = visible_gpu_count()
    total = config.WORKERS_PER_GPU * gpus
    cpu_total = config.cpu_workers(total)
    logger.info("Starting %d GPU worker process%s (%d per GPU across %d GPU%s) "
                "and %d CPU worker process%s",
                total, "" if total == 1 else "es",
                config.WORKERS_PER_GPU, gpus, "" if gpus == 1 else "s",
                cpu_total, "" if cpu_total == 1 else "es")
    pool = WorkerPool(WORKER_SPEC, config.WORKERS_PER_GPU, visible_gpus=gpus,
                      options=_worker_options(), label="gpu-worker")
    await pool.start()  # closes its workers itself if it fails
    try:
        # The CPU workers build the model's input transform from the config a
        # GPU worker reported (they load no model).
        cpu_pool = WorkerPool(CPU_WORKER_SPEC, cpu_total, cuda=False, label="cpu-worker",
                              options={"predict_transform": pool.info["predict_transform"],
                                       "cpu_threads": _cpu_threads()})
        await cpu_pool.start()
    except BaseException:
        await pool.close()
        raise
    state.pool = pool
    state.cpu_pool = cpu_pool
    state.ready = True
    state.loaded_at = time.time()
    logger.info("Pipeline ready: %d GPU worker%s, %d CPU worker%s (%s)",
                total, "" if total == 1 else "s", cpu_total, "" if cpu_total == 1 else "s",
                {k: v for k, v in pool.info.items() if k != "predict_transform"})
    try:
        yield
    finally:
        state.ready = False
        state.pool = None
        state.cpu_pool = None
        # Stopped together: each lets its running jobs finish first.
        await asyncio.gather(pool.close(), cpu_pool.close())


app = FastAPI(title="SkinTokens / TokenRig Inference Server", version="1.0",
              lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready or state.pool is None or state.cpu_pool is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    pool = state.pool.status()
    cpu_pool = state.cpu_pool.status()
    return {
        "status": "ok",
        "busy": state.busy,
        "active_jobs": state.active_jobs,
        "admitted": state.admitted,
        "max_admitted": _admission_limit(),
        "workers": pool["workers"],
        "cpu_workers": cpu_pool["workers"],
        "pool": pool,
        "cpu_pool": cpu_pool,
        "queue_depth": pool["queued"] + cpu_pool["queued"],
        "output_formats": list(config.OUTPUT_FORMATS),
    }


# Errors caused by the request itself (bad parameters or upload, a client that
# went away): logged as warnings; anything else as an error.
_CLIENT_ERRORS = (ValueError, WebSocketDisconnect)


def _is_benign_ws_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, WebSocketDisconnect):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "cannot call" in msg and "send" in msg and "close" in msg
        ) or "response already completed" in msg
    return False


class _RequestRecord:
    """One request's details: the source of its single summary log line and,
    when the server fails on it, of its dump (``dump/<date>/<request_id>/``: upload as
    received, header frame, traceback (with the client address) and trace). Create it
    in the request's handler task so the trace also collects the DEBUG lines of tasks it
    starts.
    """

    def __init__(self, request_id: str, client) -> None:
        self.request_id = request_id
        self.client = client
        self.client_addr = self._client_address(client)
        self.started = time.monotonic()
        self.status = "started"
        self.error: Optional[str] = None
        self.server_error = False
        self.traceback: Optional[str] = None
        self.raw_header: Optional[str] = None
        self.header_valid = False
        self.filename: Optional[str] = None
        self.params: Optional[dict] = None
        self.upload: Optional[bytes] = None  # kept for a dump; released on success
        self.upload_bytes: Optional[int] = None
        self.samples: Optional[int] = None
        self.output_bytes: Optional[int] = None
        self.trace = RequestTrace()
        self._trace_token = bind_trace(self.trace)

    def fail(self, exc: BaseException) -> None:
        if isinstance(exc, (asyncio.CancelledError, GenerationCancelled)):
            status = "cancelled"
        elif _is_benign_ws_disconnect(exc):
            status = "disconnected"
        else:
            status = "error"
        self.status = status
        self.error = f"{type(exc).__name__}: {exc}"
        self.server_error = status == "error" and not isinstance(exc, _CLIENT_ERRORS)
        # Formatted now, so the record does not keep the failed frames alive.
        self.traceback = "".join(traceback.format_exception(exc)) if self.server_error else None

    def reject(self, message: str) -> None:
        """A request refused with an error frame (busy, loading, malformed,
        timed out): a ``rejected`` warning line, never a dump."""
        self.status = "rejected"
        self.error = message

    @staticmethod
    def _client_address(client) -> str:
        """The client's ``ip:port`` (``-`` when no address is known)."""
        if client is None:
            return "-"
        if hasattr(client, "host"):  # Starlette Address
            return f"{client.host}:{client.port}"
        if isinstance(client, tuple) and len(client) == 2:
            return f"{client[0]}:{client[1]}"
        return str(client)

    def dump_files(self) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        if self.upload is not None:
            suffix = Path(str(self.filename or "")).suffix.lower()
            files["input" + (suffix if suffix in SUPPORTED_EXT else ".bin")] = self.upload
        if self.raw_header is not None:
            files["params.json" if self.header_valid else "params.txt"] = self.raw_header.encode("utf-8")
        body = self.traceback or self.error or ""
        files["traceback.txt"] = f"client={self.client_addr}\n\n{body}".encode("utf-8")
        files["trace.log"] = self.trace.text().encode("utf-8")
        return files

    def log_line(self, elapsed: float) -> tuple[int, str]:
        parts = [f"[{self.request_id}]", "generate", f"status={self.status}",
                 f"elapsed={elapsed:.2f}s", f"client={self.client_addr}"]
        if self.filename is not None:
            parts.append(f"filename={self.filename!r}")
        if self.upload_bytes is not None:
            parts.append(f"bytes={self.upload_bytes}")
        if self.params is not None:
            parts.append("params=" + json.dumps(self.params, ensure_ascii=False,
                                                separators=(",", ":"), default=str))
        if self.samples is not None:
            parts.append(f"samples={self.samples}")
        if self.output_bytes is not None:
            parts.append(f"output_bytes={self.output_bytes}")
        if self.error:
            parts.append("error=" + json.dumps(self.error, ensure_ascii=False))
        if self.dump_error:
            parts.append("dump_error=" + json.dumps(self.dump_error, ensure_ascii=False))
        if self.status == "done":
            level = logging.INFO
        elif self.status == "error" and self.server_error:
            level = logging.ERROR
        else:
            level = logging.WARNING
        return level, " ".join(parts)

    dump_error: Optional[str] = None

    def save(self) -> None:
        """Write the dump of a failed request, then the single summary line."""
        unbind_trace(self._trace_token)
        if self.status == "started":
            self.status = "incomplete"
        elapsed = time.monotonic() - self.started
        if self.status == "error" and self.server_error:  # unhandled server-side failures only
            try:
                save_request_dump(self.request_id, self.dump_files())
            except Exception as exc:  # The dump never fails the request.
                self.dump_error = f"failed to save request dump: {exc}"
        self.upload = None
        logger.log(*self.log_line(elapsed))


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    WebSocket protocol:
      client -> {"filename": "model.glb", "output": "rig"|"glb", ...params}
      client -> <binary: file data>
      client -> {"type": "cancel"}
      server -> {"stage": "queued"|"processing"|"done"|"cancelled"|"error", ...}
      server -> {"stage": "done", "format": "rig"|"glb", "payload_size": N, ...}
      server -> <binary: rig payload or GLB>

    ``output`` defaults to ``"rig"``; formats outside ``config.OUTPUT_FORMATS``
    (``"glb"`` unless SKINTOKENS_ENABLE_GLB_OUTPUT=1) are refused before the upload.
    """
    request_id = new_request_id()
    record = _RequestRecord(request_id, ws.client)
    cancellation = None
    generation_task = None
    receiver_task = None
    outbox = None

    await ws.accept()
    logger.debug("[%s] WebSocket connected client=%s", request_id, record.client_addr)

    async def reject(message: str) -> None:
        record.reject(message)
        await ws.send_json({"stage": "error", "message": message})
        await ws.close()

    admitted = False
    try:
        try:
            header = await asyncio.wait_for(ws.receive_text(), config.HEADER_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            await reject(f"no request header within {config.HEADER_TIMEOUT_SEC}s")
            return
        except KeyError:  # a binary frame
            await reject("expected a JSON request header before the file")
            return
        record.raw_header = header
        req = _parse_text_message(header)
        record.header_valid = True
        logger.debug("[%s] WebSocket generation request received", request_id)

        if not state.ready or state.pool is None or state.cpu_pool is None:
            await reject("model still loading")
            return

        filename = _request_filename(req)
        record.filename = filename
        if req.get("use_skeleton"):
            await reject("use_skeleton is no longer supported; inputs are meshes and "
                         "the skeleton is always generated")
            return
        # Validated before the upload is received, so a bad request costs no upload.
        # Unknown fields (e.g. the sampling knobs older clients still send) are
        # ignored. A request without a seed gets one here, so it can be reported
        # with the result and the run reproduced.
        params = GenParams(**{k: v for k, v in req.items() if k in GenParams.model_fields}).with_seed()
        record.params = params.model_dump()
        if params.output not in config.OUTPUT_FORMATS:
            await reject(_unsupported_output_message(params.output))
            return

        # Every admitted request holds its upload in this process until a
        # worker takes it, so their number is capped.
        if state.admitted >= _admission_limit():
            await reject(f"server busy: {state.admitted} requests running or queued; retry later")
            return
        state.admitted += 1
        admitted = True

        try:
            file_data = await asyncio.wait_for(ws.receive_bytes(), config.UPLOAD_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            await reject(f"file not received within {config.UPLOAD_TIMEOUT_SEC}s")
            return
        except KeyError:  # a text frame
            await reject("invalid file: expected a binary message")
            return

        record.upload = file_data
        record.upload_bytes = len(file_data)
        logger.debug("[%s] file decoded bytes=%d filename=%r params=%s",
                    request_id, len(file_data), filename, params.model_dump())

        cancellation = CancellationToken()
        outbox = _WsOutbox(ws, cancellation)

        def send_progress(percent, stage, elapsed):
            record.trace.event(f"progress={percent}% stage={stage} elapsed={elapsed}s")
            outbox.post({"stage": "processing", "step": stage, "progress": percent,
                         "elapsed_sec": elapsed, "request_id": request_id})

        async def send_queue(phase, waited_sec, queue_ahead):
            outbox.post({"stage": "queued", "phase": phase, "queued": True,
                         "waited_sec": waited_sec, "queue_ahead": queue_ahead,
                         "request_id": request_id})

        # Whether this request is to wait for a worker: to parse it, or for
        # the GPU it will need next.
        queued = state.cpu_pool.would_queue() or state.pool.would_queue()
        logger.debug("[%s] queued=%s", request_id, queued)
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
            record.status = "cancelled"
            record.error = cancellation.reason
            return

        receiver_task.cancel()
        with suppress(asyncio.CancelledError):
            await receiver_task

        results = await generation_task  # list of (payload, meta) tuples
        await outbox.flush()

        for sample_idx, (payload, _meta) in enumerate(results):
            await ws.send_json({
                "stage": "done",
                "sample_index": sample_idx,
                "num_samples": len(results),
                "seed": params.seed,
                "elapsed_sec": round(time.time() - t0, 2),
                "progress": 100,
                "request_id": request_id,
                "format": params.output,
                "payload_size": len(payload),
            })
            await ws.send_bytes(payload)
            record.output_bytes = (record.output_bytes or 0) + len(payload)

        record.samples = len(results)
        record.status = "done"
        await ws.close()

    except asyncio.CancelledError as e:
        record.fail(e)
        raise
    except (WebSocketDisconnect, GenerationCancelled) as e:
        record.fail(e)
    except Exception as e:
        record.fail(e)
        try:
            if outbox is not None:
                await outbox.flush()
            await ws.send_json({"stage": "error", "message": _client_error_message(e, request_id)})
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
        if admitted:
            state.admitted -= 1
        record.save()


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
    parser.add_argument("--workers", type=int, default=config.WORKERS_PER_GPU,
                        help="GPU worker processes per GPU; SKINTOKENS_WORKERS")
    parser.add_argument("--batch-slots", type=int, default=config.BATCH_SLOTS,
                        choices=(0, 1, 2, 3, 4),
                        help="requests each GPU worker decodes at once (0: no "
                             "batching); SKINTOKENS_BATCH_SLOTS")
    parser.add_argument("--cpu-workers", type=int, default=config.CPU_WORKERS,
                        help="CPU worker processes for parsing and export "
                             "(0 = twice the GPU workers); SKINTOKENS_CPU_WORKERS")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.cpu_workers < 0:
        parser.error("--cpu-workers must not be negative")
    config.MODEL_CKPT = args.model_ckpt
    config.HF_PATH = args.hf_path
    config.WORKERS_PER_GPU = args.workers
    config.CPU_WORKERS = args.cpu_workers
    config.BATCH_SLOTS = args.batch_slots

    import uvicorn

    class RequestNoiseFilter(logging.Filter):
        """Drops uvicorn's per-connection lines: each request logs one summary."""

        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return not ('"GET /health' in message
                        or message.startswith("WebSocket /ws/")
                        or message in ("connection open", "connection closed"))

    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(RequestNoiseFilter())

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"] = {
        name: {**formatter, "fmt": f"%(asctime)s.%(msecs)03d {formatter['fmt']}",
               "datefmt": LOG_DATE_FORMAT}
        for name, formatter in uvicorn.config.LOGGING_CONFIG["formatters"].items()
    }
    # One gateway process (uvicorn workers=1): the model workers are the pools'.
    # No per-message deflate: uploads and rigged GLBs are mostly float data that
    # barely compresses, and zlib-ing tens of MB (on the event loop, stalling
    # every other connection) cost ~2s per request on a 24 MB upload.
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=MAX_UPLOAD_BYTES + 64 * 1024, ws_per_message_deflate=False,
                log_config=log_config)


if __name__ == "__main__":
    main()
