"""Worker process: owns one copy of the model and runs its requests.

Started by :class:`src.server.pool.WorkerPool` with the ``spawn`` method (CUDA
cannot be used across a fork). It talks to the gateway over two one-way pipes:

gateway → worker
    ``("job", job_id, params, upload)``, ``("cancel", job_id)`` and ``("stop",)``.
worker → gateway
    ``("ready", info)`` or ``("failed", traceback)`` once, after loading;
    then per job any number of ``("progress", job_id, event)``, ended by
    ``("result", job_id, value)`` or ``("error", job_id, error)``.

Jobs run on the process's main thread (the thread that loaded the model and
captured any CUDA graphs); a reader thread takes messages off the jobs pipe so
that a ``cancel`` reaches the running job while it works.

A worker runs one job at a time, unless its spec has a ``serve`` loop and its
``load`` reported ``slots`` (> 0) in its info: the gateway then sends it up to
that many jobs at once, and ``serve`` runs them together (continuous batching,
:mod:`src.server.pipeline`), taking jobs from a :class:`JobSource`.

This module stays free of model imports: only the spec a worker runs decides
what it loads, so tests can run workers with a fixture spec and no model.
"""
from __future__ import annotations

import importlib
import logging
import os
import pickle
import queue
import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.server.logs import configure_logging

ProgressCallback = Callable[[dict], None]


class GenerationCancelled(Exception):
    """Raised when a generation request has been cancelled."""


class Cancellation:
    """A job's cancellation flag, set by the reader thread, checked by the job."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = "generation cancelled"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "generation cancelled") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise GenerationCancelled(self.reason)


@dataclass(frozen=True)
class ServiceSpec:
    """What a worker process runs.

    ``load(options)`` loads the model into this process and returns what the
    gateway needs to know about it (sent once, when the worker is ready).
    ``run(params, upload, cancellation, progress)`` serves one request.
    ``idle()`` (optional) runs in the worker after each request's reply has
    been sent, for housekeeping that must not delay that reply.
    ``serve(source)`` (optional) replaces the one-job-at-a-time loop when
    ``load`` reports ``slots`` in its info: it runs until
    :meth:`JobSource.next` raises :class:`Stopped`, taking jobs and answering
    each through its :class:`Job`.
    """
    name: str
    load: Callable[[dict], dict]
    run: Callable[[dict, bytes, Cancellation, ProgressCallback], Any]
    idle: Optional[Callable[[], None]] = None
    serve: Optional[Callable[["JobSource"], None]] = None


class _Sender:
    """Thread-safe sending on the worker → gateway pipe."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, message: tuple) -> None:
        with self._lock:
            self._conn.send(message)


def pack_exception(exc: BaseException) -> dict:
    """An exception as it crosses the pipe: pickled when it round-trips, else
    its type and message; always with its formatted traceback."""
    text = "".join(traceback.format_exception(exc))
    try:
        data = pickle.dumps(exc)
        pickle.loads(data)
    except Exception:
        data = None
    return {"exception": data, "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc), "traceback": text}


def resolve_spec(reference: str) -> ServiceSpec:
    """``"package.module:NAME"`` → that attribute."""
    module, _, name = reference.partition(":")
    return getattr(importlib.import_module(module), name)


class _Inbox:
    """Reads the jobs pipe on a thread: jobs queue up for the main thread,
    cancels reach their job's :class:`Cancellation` at once."""

    STOP = object()

    def __init__(self, conn) -> None:
        self._conn = conn
        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._tokens: dict[int, Cancellation] = {}
        # A cancel can overtake its job: the gateway sends both from threads.
        self._early_cancels: set[int] = set()
        threading.Thread(target=self._read, name="jobs-reader", daemon=True).start()

    def _read(self) -> None:
        try:
            while True:
                message = self._conn.recv()
                kind = message[0]
                if kind == "job":
                    token = Cancellation()
                    with self._lock:
                        if message[1] in self._early_cancels:
                            self._early_cancels.discard(message[1])
                            token.cancel("client requested cancellation")
                        self._tokens[message[1]] = token
                    self._jobs.put((message, token))
                elif kind == "cancel":
                    with self._lock:
                        token = self._tokens.get(message[1])
                        if token is None:
                            self._early_cancels.add(message[1])
                    if token is not None:
                        token.cancel(message[2] if len(message) > 2 else "client requested cancellation")
                elif kind == "stop":
                    break
        except (EOFError, OSError):
            pass  # The gateway is gone.
        self._jobs.put(self.STOP)

    def next(self, block: bool = True):
        """The next ``(message, token)`` or :attr:`STOP`; None if *block* is
        false and nothing is waiting."""
        if block:
            return self._jobs.get()
        try:
            return self._jobs.get_nowait()
        except queue.Empty:
            return None

    def done(self, job_id: int) -> None:
        with self._lock:
            self._tokens.pop(job_id, None)


class Stopped(Exception):
    """The gateway stopped the worker (or is gone): ``serve`` should return."""


class Job:
    """One job as a ``serve`` loop holds it; answered once, by :meth:`reply`
    or :meth:`fail`."""

    def __init__(self, source: "JobSource", job_id: int, params: dict, upload: Any,
                 cancellation: Cancellation) -> None:
        self._source = source
        self.id = job_id
        self.params = params
        self.upload = upload
        self.cancellation = cancellation
        self.answered = False

    def progress(self, event: dict) -> None:
        self._source._sender.send(("progress", self.id, event))

    def reply(self, value: Any) -> None:
        self._answer(("result", self.id, value))

    def fail(self, exc: BaseException) -> None:
        self._answer(("error", self.id, pack_exception(exc)))

    def _answer(self, reply: tuple) -> None:
        if self.answered:
            return
        self.answered = True
        self.upload = None
        self._source._inbox.done(self.id)
        _send_reply(self._source._sender, reply)


class JobSource:
    """Where a ``serve`` loop takes its jobs from."""

    def __init__(self, inbox: _Inbox, sender: _Sender) -> None:
        self._inbox = inbox
        self._sender = sender

    def next(self, block: bool) -> Optional[Job]:
        """The next job; None if *block* is false and none is waiting.
        Raises :class:`Stopped` once the gateway has stopped the worker."""
        item = self._inbox.next(block)
        if item is None:
            return None
        if item is _Inbox.STOP:
            raise Stopped()
        (_, job_id, params, upload), cancellation = item
        return Job(self, job_id, params, upload, cancellation)


def _send_reply(sender: _Sender, reply: tuple) -> bool:
    """Send a job's reply; False if the gateway is gone."""
    try:
        sender.send(reply)
    except (EOFError, OSError):
        return False
    except Exception as exc:  # A result that cannot be pickled.
        sender.send(("error", reply[1], pack_exception(exc)))
    return True


def _pin_worker_gpu(gpu) -> None:
    """Restrict this worker to its assigned physical GPU, before CUDA loads.

    *gpu* is the physical index the gateway pinned this worker to (``None``
    when the pool is not spreading over GPUs, ``""`` for a CPU-only worker,
    which then sees no GPU). Pinning ``CUDA_VISIBLE_DEVICES`` to it makes that
    GPU the only one this process sees, so the pipeline's ``cuda`` device
    lands on it. Called at the very start of the process, before any CUDA
    context exists.
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def main(spec_reference: str, jobs, events, options: dict, gpu=None) -> None:
    """Process entry point. *jobs* is read, *events* written; *gpu* is the
    physical index the pool pinned this worker to (``None`` for a
    single-GPU pool)."""
    _pin_worker_gpu(gpu)
    # Ctrl+C in a terminal reaches the whole process group; the gateway decides
    # when workers stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Each worker gets its share of the CPU so N workers do not oversubscribe
    # it (set before torch/numpy are imported by the spec).
    threads = options.get("cpu_threads")
    if threads:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(variable, str(threads))
    configure_logging()
    log = logging.getLogger("skintokens.worker")
    sender = _Sender(events)

    try:
        spec = resolve_spec(spec_reference)
        info = spec.load(options) or {}
    except BaseException:
        log.exception("worker failed to load")
        sender.send(("failed", traceback.format_exc()))
        return
    batched = spec.serve is not None and bool(info.get("slots"))
    if not batched:
        info.pop("slots", None)  # the gateway sends one job at a time
    sender.send(("ready", {**info, "pid": os.getpid()}))

    inbox = _Inbox(jobs)
    if batched:
        try:
            spec.serve(JobSource(inbox, sender))
        except Stopped:
            pass
        return
    while True:
        item = inbox.next()
        if item is _Inbox.STOP:
            return
        (_, job_id, params, upload), cancellation = item

        def progress(event, job_id=job_id):
            sender.send(("progress", job_id, event))

        try:
            reply = ("result", job_id, spec.run(params, upload, cancellation, progress))
        except BaseException as exc:  # Never leaves the loop: fails this request only.
            reply = ("error", job_id, pack_exception(exc))
        finally:
            inbox.done(job_id)
        del upload
        if not _send_reply(sender, reply):
            return
        if spec.idle is not None:
            try:
                spec.idle()
            except Exception:  # Housekeeping never fails the worker.
                log.exception("idle hook failed")
