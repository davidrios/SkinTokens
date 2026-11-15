"""Gateway side of the worker processes.

A :class:`WorkerPool` runs a fixed number of worker processes
(:mod:`src.server.worker`), each owning a copy of the model and running one
request at a time, so up to ``size`` requests run in parallel: their parsing,
GPU inference and export never wait on one another. The pool is sized as
*per-GPU workers* times the number of visible GPUs: workers are pinned to the
cards round-robin (each sees its one GPU as ``cuda:0``). Requests wait for a
free worker in FIFO order; a waiting request gets a once-per-second queue
heartbeat.

A request can be cancelled while it waits (it leaves the queue) or while it
runs (the worker is told to stop it and is free again once it has).

A worker that exits (a crash, a fatal CUDA error, the OOM killer) fails only
the request it was running; the pool starts a replacement on the same GPU.

A pool with ``cuda=False`` runs CPU-only workers, which see no GPU at all
(the server's CPU stages, :mod:`src.server.stages`). A request submitted with
``urgent=True`` waits ahead of all non-urgent ones (in FIFO order among the
urgent), so a request's last stage is not stuck behind new requests' first.

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


_F_SETPIPE_SZ = 1031  # Linux fcntl.F_SETPIPE_SZ (exposed by Python only from 3.10 on Linux)


def _enlarge_pipe(conn, size: int = 1 << 20) -> None:
    """Grow a pipe's kernel buffer (Linux; capped by /proc/sys/fs/pipe-max-size).

    Best effort: elsewhere (Windows named pipes) or when refused, nothing changes.
    """
    try:
        import fcntl
    except ImportError:
        return
    with suppress(OSError, ValueError, AttributeError):
        fcntl.fcntl(conn.fileno(), getattr(fcntl, "F_SETPIPE_SZ", _F_SETPIPE_SZ), size)


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
        # Uploads and results are tens of MB; a pipe's default 64 KiB buffer
        # turns each into ~1000 sender/receiver hand-offs.
        _enlarge_pipe(self.jobs)
        _enlarge_pipe(self.events)
        self.process = ctx.Process(
            target=worker_module.main,
            args=(pool.spec, jobs_read, events_write, pool.options,
                  pool._gpu_for_slot(slot)),
            name=f"skintokens-{pool.label}-{slot}",
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
        return f"{self.pool.label} {self.slot} (pid {self.process.pid})"

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
        # This thread alone reaps the process. Concurrent Process.join()/poll()
        # calls can race on waitpid and temporarily report an exited child as
        # alive. EOF can also precede slow model/compiler finalizers: during
        # shutdown close() owns the deadline and escalation, not this reader.
        if not self.pool._closing:
            self.process.join(timeout=5)
            if self.process.is_alive() and not self.pool._closing:
                logger.error("%s closed its pipe but did not exit; stopping it", self.name)
                self.process.kill()
        self.process.join()
        self.events.close()
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
    """Worker processes running *spec*, spread across the visible GPUs.

    *per_gpu* is the number of workers to run **on each** visible GPU; the pool
    starts ``per_gpu * visible_gpus`` workers and pins them to the GPUs
    round-robin (each worker sees its one card as ``cuda:0``).

    *spec* names the worker's :class:`src.server.worker.ServiceSpec` as
    ``"module:NAME"``; *options* reach every worker's ``load``.

    With ``cuda=False`` the workers see no GPU (``CUDA_VISIBLE_DEVICES`` is
    empty) and *visible_gpus* must stay 1: *per_gpu* is then the pool size.
    *label* names the workers in logs and process names.
    """

    def __init__(self, spec: str, per_gpu: int, *, visible_gpus: int = 1,
                 options: Optional[dict] = None, restart_delay: float = 5.0,
                 cuda: bool = True, label: str = "worker") -> None:
        per_gpu = int(per_gpu)
        visible_gpus = max(1, int(visible_gpus))
        if per_gpu < 1:
            raise ValueError(f"the pool needs at least one worker per GPU, got {per_gpu}")
        if not cuda and visible_gpus != 1:
            raise ValueError("a CPU-only pool is not spread over GPUs")
        self.spec = spec
        self.cuda = cuda
        self.label = label
        self.per_gpu = per_gpu
        self.gpus = visible_gpus
        # One worker per (GPU, per-GPU slot): per_gpu workers on each card.
        self.size = per_gpu * visible_gpus
        self.options = dict(options or {})
        self.restart_delay = restart_delay
        self.info: dict = {}
        self.restarts = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._workers: list[_Worker] = []
        self._idle: deque[_Worker] = deque()
        self._waiters: deque[asyncio.Future] = deque()
        # The waiters of urgent requests, all ahead of the others in _waiters.
        self._urgent: set[asyncio.Future] = set()
        self._ids = itertools.count(1)
        self._closing = False
        self._respawns: set[asyncio.Task] = set()

    def _gpu_for_slot(self, slot: int) -> Optional[int | str]:
        """The physical GPU *slot* runs on, ``None`` for a single-GPU pool, or
        ``""`` (no GPU visible) for a CPU-only pool.

        Slots fill the cards round-robin (slot ``g`` of each GPU), so a pool of
        *per_gpu* workers over *gpus* cards gives card *k* the slots
        ``k, k + gpus, k + 2 gpus, ...``.
        """
        if not self.cuda:
            return ""
        if self.gpus <= 1:
            return None
        return slot % self.gpus

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
        if self.cuda:
            logger.info("%s pool ready: %d worker%s across %d GPU%s (%d per GPU) in %.1fs",
                        self.label, self.size, "" if self.size == 1 else "s",
                        self.gpus, "" if self.gpus == 1 else "s",
                        self.per_gpu, time.monotonic() - started)
        else:
            logger.info("%s pool ready: %d CPU-only worker%s in %.1fs", self.label,
                        self.size, "" if self.size == 1 else "s", time.monotonic() - started)

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
            def wait(pending: list[_Worker], seconds: float) -> list[_Worker]:
                deadline = time.monotonic() + seconds
                for worker in pending:
                    worker.reader.join(max(0.0, deadline - time.monotonic()))
                return [worker for worker in pending if worker.reader.is_alive()]

            pending = wait(workers, timeout)
            for worker in pending:
                logger.warning("%s exceeded shutdown timeout; terminating it", worker.name)
                worker.process.terminate()
            pending = wait(pending, 5)
            for worker in pending:
                logger.warning("%s did not terminate; killing it", worker.name)
                worker.process.kill()
            pending = wait(pending, 5)
            if pending:
                raise RuntimeError("worker processes did not exit after being killed")

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

    def _enqueue(self, waiter: asyncio.Future, urgent: bool) -> None:
        if not urgent:
            self._waiters.append(waiter)
            return
        position = 0
        while position < len(self._waiters) and self._waiters[position] in self._urgent:
            position += 1
        self._urgent.add(waiter)
        self._waiters.insert(position, waiter)

    async def _acquire(self, queue_callback: Optional[QueueCallback], queue_phase: str,
                       cancelled: Optional[asyncio.Event], urgent: bool = False) -> _Worker:
        if self._closing:
            raise RuntimeError("the server is shutting down")
        if cancelled is not None and cancelled.is_set():
            raise GenerationCancelled("generation cancelled")
        if self._idle and not self._waiters:
            return self._idle.popleft()
        waiter = self._loop.create_future()
        self._enqueue(waiter, urgent)
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
            self._urgent.discard(waiter)
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
                     cancel_reason: Callable[[], str] = lambda: "generation cancelled",
                     urgent: bool = False) -> Any:
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

        An *urgent* request waits ahead of every non-urgent one.
        """
        started = False
        while True:
            worker = await self._acquire(queue_callback, queue_phase, cancelled, urgent)
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
            "per_gpu": self.per_gpu,
            "gpus": self.gpus,
            "ready": len(ready),
            "busy": sum(1 for w in ready if w.job is not None),
            "idle": len(self._idle),
            "queued": sum(1 for waiter in self._waiters if not waiter.done()),
            "restarts": self.restarts,
        }
