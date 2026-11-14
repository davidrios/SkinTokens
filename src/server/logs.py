"""Logging and failure dumps of the SkinTokens service (no torch imports).

* One summary line per request at INFO (``serve.py``); stage-by-stage details
  are DEBUG. ``SKINTOKENS_LOG_LEVEL`` filters only what reaches the log: a
  failed request's dump still gets every DEBUG line (:class:`RequestTrace`).
* A failed request leaves ``dump/<date>/<request_id>/`` under
  ``SKINTOKENS_OUTPUT_ROOT`` with its upload as received, its header frame,
  traceback and trace.
* Daily log files are written by :mod:`src.server.log_archive`, which wraps the
  server command in the container.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from contextlib import suppress
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(processName)s] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAMES = ("skintokens.serve", "skintokens.worker")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def output_root() -> Path:
    root = os.environ.get("SKINTOKENS_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "server"))
    return Path(root).expanduser().resolve()


def configure_logging() -> None:
    """Route the service loggers to stderr at ``SKINTOKENS_LOG_LEVEL`` (INFO).

    The loggers themselves pass every level so a request's trace sees DEBUG.
    Idempotent: also called by each worker process.
    """
    level = os.environ.get("SKINTOKENS_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    for name in LOGGER_NAMES:
        log = logging.getLogger(name)
        for old in list(log.handlers):
            log.removeHandler(old)
        log.addHandler(stream)
        if name == "skintokens.serve":
            log.addHandler(_TraceHandler())
        log.setLevel(logging.DEBUG)
        log.propagate = False


# --------------------------------------------------------------------------- #
# Request trace
# --------------------------------------------------------------------------- #
class RequestTrace:
    """Timestamped lines of one request, kept in memory for its failure dump
    (the newest :attr:`MAX_LINES`)."""

    MAX_LINES = 5000

    def __init__(self) -> None:
        self._lines: deque[str] = deque(maxlen=self.MAX_LINES)
        self._total = 0
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self._total += 1
            self._lines.append(line)

    def event(self, message: str) -> None:
        now = time.time()
        stamp = time.strftime(LOG_DATE_FORMAT, time.localtime(now))
        self.add(f"{stamp}.{int(now % 1 * 1000):03d} EVENT {message}")

    def text(self) -> str:
        with self._lock:
            lines = list(self._lines)
            dropped = self._total - len(lines)
        if dropped:
            lines.insert(0, f"... {dropped} earlier lines dropped ...")
        return "".join(line + "\n" for line in lines)


# The trace of the request being handled; tasks the request starts inherit it.
_current_trace: ContextVar[Optional[RequestTrace]] = ContextVar("skintokens_trace", default=None)


def bind_trace(trace: Optional[RequestTrace]) -> Token:
    return _current_trace.set(trace)


def unbind_trace(token: Token) -> None:
    with suppress(ValueError):  # Token from another context.
        _current_trace.reset(token)


class _TraceHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        trace = _current_trace.get()
        if trace is not None:
            try:
                trace.add(self.format(record))
            except Exception:
                self.handleError(record)


# --------------------------------------------------------------------------- #
# Failure dumps
# --------------------------------------------------------------------------- #
def request_day(timestamp: float) -> str:
    """``YYYY-MM-DD`` in local time (like the daily log files; set ``TZ``)."""
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def safe_name(name: str, fallback: str = "file") -> str:
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(name or fallback).name).strip("._")
    return base or fallback


def save_request_dump(request_id: str, files: dict[str, bytes], *,
                      timestamp: Optional[float] = None) -> Path:
    """Write a failed request's files to ``dump/<date>/<request_id>/``."""
    timestamp = time.time() if timestamp is None else timestamp
    directory = output_root() / "dump" / request_day(timestamp) / safe_name(request_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = directory / safe_name(name)
        path.write_bytes(data)
        os.utime(path, (timestamp, timestamp))
    os.utime(directory, (timestamp, timestamp))
    return directory
