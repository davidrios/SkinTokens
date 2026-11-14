"""Gateway side of the worker processes.

A :class:`WorkerPool` runs a fixed number of worker processes
(:mod:`src.server.worker`), each owning a copy of the model and running one
request at a time, so up to ``size`` requests run in parallel: their parsing,
GPU inference and export never wait on one another. Requests wait for a free
worker in FIFO order; a waiting request gets a once-per-second queue heartbeat.

A request can be cancelled while it waits (it leaves the queue) or while it
runs (the worker is told to stop it and is free again once it has).

A worker that exits (a crash, a fatal CUDA error, the OOM killer) fails only
the request it was running; the pool starts a replacement.

Adapted from the AnyCross server's worker pool.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import multiprocessing
import pickle
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from src.server import worker as worker_module
from src.server.worker import GenerationCancelled

logger = logging.getLogger("skintokens.serve")

QueueCallback = Callable[[str, float, int], Awaitable[None]]


class WorkerCrashed(RuntimeError):
    """The worker process running a request exited before answering."""


class RemoteTraceback(Exception):
    """The traceback of an exception raised in a worker, as its ``__cause__``."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return f'\n"""\n{self.text.rstrip()}\n"""'


class RemoteError(RuntimeError):
    """An exception from a worker that could not be recreated in the gateway."""


def unpack_exception(error: dict) -> BaseException:
    """The exception a worker reported (:func:`src.server.worker.pack_exception`),
    with the worker's traceback chained as its cause."""
    exc: Optional[BaseException] = None
    if error.get("exception") is not None:
        with suppress(Exception):
            exc = pickle.loads(error["exception"])
    if not isinstance(exc, BaseException):
        exc = RemoteError(f"{error.get('type')}: {error.get('message')}")
    exc.__cause__ = RemoteTraceback(error.get("traceback") or "")
    return exc


@dataclass
class _Job:
    id: int
    future: asyncio.Future
    on_progress: Optional[Callable[[dict], Any]]
    cancel_sent: bool = False


class _Worker:
    """One worker process and the thread reading its events."""

    def __init__(self, pool: "WorkerPool", slot: int) -> None:
        self.pool = pool
        self.slot = slot
        loop = pool._loop
        ctx = multiprocessing.get_context("spawn")
        jobs_read, self.jobs = ctx.Pipe(duplex=False)
        self.events, events_write = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=worker_module.main,
            args=(pool.spec, jobs_read, events_write, pool.options),
            name=f"skintokens-worker-{slot}",
        )
        self.process.start()
        # Only the child keeps these ends, so its exit closes the pipes.
        jobs_read.close()
        events_write.close()
        self.ready: asyncio.Future = loop.create_future()
        self.info: dict = {}
        self.alive = True
        self.job: Optional[_Job] = None
        self.jobs_done = 0
        # Jobs and cancels are sent from pool threads, "stop" from the loop.
        self._send_lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, name=f"{self.process.name}-events",
                                       daemon=True)
        self.reader.start()

    @property
    def name(self) -> str:
        return f"worker {self.slot} (pid {self.process.pid})"

    def _read(self) -> None:
        loop = self.pool._loop
        try:
            while True:
                message = self.events.recv()
                loop.call_soon_threadsafe(self._on_message, message)
        except (EOFError, OSError):
            pass
        except Exception:
            # A message that cannot be unpickled leaves the stream unusable:
            # the worker can no longer be heard, so it must not keep running.
            logger.exception("%s sent an unreadable message; stopping it", self.name)
            self.process.kill()
        # The pipe closes just before the process is reaped; wait for its exit
        # code, off the event loop.
        self.process.join(timeout=5)
        if self.process.is_alive():
            logger.error("%s closed its pipe but did not exit; stopping it", self.name)
            self.process.kill()
            self.process.join(timeout=5)
        with suppress(RuntimeError):  # The loop is closed.
            loop.call_soon_threadsafe(self._on_exit)

    def _on_message(self, message: tuple) -> None:
        kind = message[0]
        if kind == "ready":
            self.info = message[1] or {}
            if not self.ready.done():
                self.ready.set_result(self.info)
        elif kind == "failed":
            if not self.ready.done():
                self.ready.set_exception(RuntimeError(f"{self.name} failed to load:\n{message[1]}"))
        elif kind == "progress":
            job = self.job
            if job is not None and message[1] == job.id and job.on_progress is not None:
                try:
                    job.on_progress(message[2])
                except Exception:
                    logger.exception("progress callback failed")
        elif kind in ("result", "error"):
            job = self.job
            if job is None or message[1] != job.id:
                return
            self.job = None
            self.jobs_done += 1
            if not job.future.done():
                if kind == "result":
                    job.future.set_result(message[2])
                else:
                    job.future.set_exception(unpack_exception(message[2]))
            self.pool._release(self)

    def _on_exit(self) -> None:
        if not self.alive:
            return
        self.alive = False
        code = self.process.exitcode
        if not self.ready.done():
            self.ready.set_exception(WorkerCrashed(f"{self.name} exited while loading (exit code {code})"))
        job, self.job = self.job, None
        if job is not None and not job.future.done():
            job.future.set_exception(WorkerCrashed(
                f"{self.name} exited (exit code {code}) while running this request"))
        self.pool._on_worker_exit(self, code)

    def send(self, message: tuple) -> None:
        with self._send_lock:
            self.jobs.send(message)

    def try_send(self, message: tuple) -> None:
        with suppress(OSError, ValueError, EOFError):
            self.send(message)

    def stop(self) -> None:
        self.try_send(("stop",))


def _retrieve(future: asyncio.Future) -> None:
    """Mark a job's outcome as retrieved when its request no longer waits."""
    if not future.cancelled():
        future.exception()


async def _first(future: asyncio.Future, event: Optional[asyncio.Event]) -> None:
    """Wait until *future* is done or *event* is set, without cancelling *future*."""
    if event is None:
        await asyncio.wait((future,))
        return
    if event.is_set():
        return
    waiter = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait((future, waiter), return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()


class WorkerPool:
    """A fixed number of worker processes running *spec*.

    *spec* names the worker's :class:`src.server.worker.ServiceSpec` as
    ``"module:NAME"``; *options* reach every worker's ``load``.
    """

    def __init__(self, spec: str, size: int, *, options: Optional[dict] = None,
                 restart_delay: float = 5.0) -> None:
        if size < 1:
            raise ValueError(f"the pool needs at least one worker, got {size}")
        self.spec = spec
        self.size = size
        self.options = dict(options or {})
        self.restart_delay = restart_delay
        self.info: dict = {}
        self.restarts = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._workers: list[_Worker] = []
        self._idle: deque[_Worker] = deque()
        self._waiters: deque[asyncio.Future] = deque()
        self._ids = itertools.count(1)
        self._closing = False
        self._respawns: set[asyncio.Task] = set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start every worker and wait until all have loaded their model."""
        self._loop = asyncio.get_running_loop()
        started = time.monotonic()
        self._workers = [_Worker(self, slot) for slot in range(self.size)]
        results = await asyncio.gather(*(w.ready for w in self._workers), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            await self.close()
            raise failures[0]
        self.info = self._workers[0].info
        for worker in self._workers:
            self._release(worker)
        logger.info("worker pool ready: %d worker%s in %.1fs", self.size,
                    "" if self.size == 1 else "s", time.monotonic() - started)

    async def close(self, timeout: float = 30.0) -> None:
        """Stop the workers; running jobs finish first, up to *timeout* seconds."""
        self._closing = True
        for task in self._respawns:
            task.cancel()
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_exception(RuntimeError("the server is shutting down"))
        workers = list(self._workers)
        for worker in workers:
            worker.stop()

        def join() -> None:
            deadline = time.monotonic() + timeout
            for worker in workers:
                worker.process.join(max(0.0, deadline - time.monotonic()))
            for worker in workers:
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join(5)
                if worker.process.is_alive():
                    worker.process.kill()
                    worker.process.join(5)

        await asyncio.to_thread(join)
        for worker in workers:
            with suppress(OSError):
                worker.jobs.close()

    def _on_worker_exit(self, worker: _Worker, code: Optional[int]) -> None:
        with suppress(ValueError):
            self._idle.remove(worker)
        if self._closing or self._workers[worker.slot] is not worker:
            return
        if worker.ready.exception() is not None:
            return  # It never loaded: start() or _replace() reports and handles that.
        logger.error("%s exited (exit code %s); starting a replacement", worker.name, code)
        task = self._loop.create_task(self._replace(worker.slot))
        self._respawns.add(task)
        task.add_done_callback(self._respawns.discard)

    async def _replace(self, slot: int) -> None:
        delay = 0.0
        while not self._closing:
            if delay:
                await asyncio.sleep(delay)
            worker = _Worker(self, slot)
            self._workers[slot] = worker
            try:
                await worker.ready
            except Exception as exc:
                delay = min(60.0, max(self.restart_delay, delay * 2))
                logger.error("%s failed to start, retrying in %.0fs: %s", worker.name, delay, exc)
                continue
            if self._closing:
                worker.stop()
                return
            self.restarts += 1
            logger.info("%s replaced", worker.name)
            self._release(worker)
            return

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _release(self, worker: _Worker) -> None:
        """Hand a free worker to the longest-waiting request, or park it."""
        if not worker.alive or self._closing:
            return
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(worker)
                return
        self._idle.append(worker)

    def would_queue(self) -> bool:
        """Whether a request submitted now would wait for a worker."""
        return not self._idle or bool(self._waiters)

    async def _acquire(self, queue_callback: Optional[QueueCallback], queue_phase: str,
                       cancelled: Optional[asyncio.Event]) -> _Worker:
        if self._closing:
            raise RuntimeError("the server is shutting down")
        if cancelled is not None and cancelled.is_set():
            raise GenerationCancelled("generation cancelled")
        if self._idle and not self._waiters:
            return self._idle.popleft()
        waiter = self._loop.create_future()
        self._waiters.append(waiter)
        heartbeat = None
        if queue_callback is not None:
            started = time.monotonic()

            async def beat() -> None:
                while True:
                    await asyncio.sleep(1.0)
                    try:
                        ahead = self._waiters.index(waiter)
                    except ValueError:
                        return
                    await queue_callback(queue_phase, round(time.monotonic() - started, 2), ahead)

            heartbeat = asyncio.create_task(beat())

        def leave() -> None:
            """Leave the queue; a worker handed over meanwhile goes to the next."""
            if waiter.done():
                if not waiter.cancelled() and waiter.exception() is None:
                    self._release(waiter.result())
            else:
                waiter.cancel()
                with suppress(ValueError):
                    self._waiters.remove(waiter)

        try:
            await _first(waiter, cancelled)
        except asyncio.CancelledError:
            leave()
            raise
        finally:
            # Not awaited: an await here could lose a worker already handed over.
            if heartbeat is not None:
                heartbeat.cancel()
        if cancelled is not None and cancelled.is_set():
            leave()
            raise GenerationCancelled("generation cancelled")
        return waiter.result()

    def _cancel(self, worker: _Worker, job: _Job, reason: str) -> None:
        """Ask the worker to stop *job* (off the loop: a job send may hold the pipe)."""
        if job.cancel_sent or job.future.done() or worker.job is not job:
            return
        job.cancel_sent = True
        self._loop.run_in_executor(None, worker.try_send, ("cancel", job.id, reason))

    async def submit(self, params: dict, upload: bytes, *,
                     on_start: Optional[Callable[[], Awaitable[None]]] = None,
                     on_progress: Optional[Callable[[dict], Any]] = None,
                     queue_callback: Optional[QueueCallback] = None,
                     queue_phase: str = "worker",
                     cancelled: Optional[asyncio.Event] = None,
                     cancel_reason: Callable[[], str] = lambda: "generation cancelled") -> Any:
        """Run one request on the next free worker and return its result.

        *on_start* is awaited when the request gets a worker; *on_progress*
        receives the job's progress events on the event loop, all of them
        before this returns. The worker's exception is re-raised here, with its
        traceback as ``__cause__``.

        Setting *cancelled* removes a waiting request from the queue, or asks
        the worker to stop a running one; either way this then raises
        :class:`GenerationCancelled`, a running job only once the worker has
        really stopped it (so its worker is free again). Cancelling the call
        itself also asks the worker to stop the job.
        """
        started = False
        while True:
            worker = await self._acquire(queue_callback, queue_phase, cancelled)
            if not started:
                started = True
                try:
                    if on_start is not None:
                        await on_start()
                except BaseException:
                    self._release(worker)
                    raise
            # The worker may have exited since it was handed over (during
            # on_start, or before this task resumed); its exit was handled
            # without this job, and nothing was sent to it: take the next one.
            # No await from this check until the job is attached, so a later
            # exit always finds the job.
            if worker.alive:
                break
            logger.debug("%s exited before its job was sent; taking the next worker", worker.name)
        job = _Job(next(self._ids), self._loop.create_future(), on_progress)
        job.future.add_done_callback(_retrieve)
        worker.job = job
        sending = asyncio.ensure_future(
            asyncio.to_thread(worker.send, ("job", job.id, params, upload)))
        sending.add_done_callback(lambda task: self._sent(worker, job, task))
        try:
            await _first(job.future, cancelled)
            if not job.future.done():
                self._cancel(worker, job, cancel_reason())
            return await asyncio.shield(job.future)
        except asyncio.CancelledError:
            self._cancel(worker, job, "server request task cancelled")
            raise

    def _sent(self, worker: _Worker, job: _Job, task: asyncio.Future) -> None:
        exc = None if task.cancelled() else task.exception()
        if exc is None:
            return
        if isinstance(exc, OSError):
            # The worker exited. Its exit fails the job it finds attached; this
            # covers an exit already handled without it.
            if not worker.alive and not job.future.done():
                job.future.set_exception(WorkerCrashed(
                    f"{worker.name} exited before it received this request"))
            return
        # Nothing reached the worker (the job could not be pickled).
        if worker.job is job:
            worker.job = None
            if not job.future.done():
                job.future.set_exception(exc)
            self._release(worker)

    # ── status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        ready = [w for w in self._workers
                 if w.alive and w.ready.done() and not w.ready.exception()]
        return {
            "workers": self.size,
            "ready": len(ready),
            "busy": sum(1 for w in ready if w.job is not None),
            "idle": len(self._idle),
            "queued": sum(1 for waiter in self._waiters if not waiter.done()),
            "restarts": self.restarts,
        }
