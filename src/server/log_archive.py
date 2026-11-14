#!/usr/bin/env python3
"""Run a command and archive everything it prints, one file per day.

    python -m src.server.log_archive <command> [args...]

The container runs the server under this wrapper. Its stdout and stderr still reach this process's stdout and
stderr unchanged, so ``docker logs`` sees exactly what it did before, and are
also appended to ``$SKINTOKENS_OUTPUT_ROOT/logs/skintokens-YYYY-MM-DD.log``, named by
the local date. The file only switches at a line boundary, so no line is split
across two days.

SIGTERM and SIGINT are forwarded only to the command, which owns the shutdown
of its workers and their resources; the wrapper exits with
the command's exit code. Archiving never stops the command: if the log
directory cannot be written, the wrapper says so once on stderr and carries on.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PREFIX = "skintokens"


def log_dir() -> Path:
    # Same default as src.server.logs.output_root().
    root = os.environ.get("SKINTOKENS_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "server"))
    return Path(root).expanduser().resolve() / "logs"


class DailyLog:
    """Append bytes to ``<directory>/<prefix>-YYYY-MM-DD.log`` for the current day."""

    def __init__(self, directory: Path, *, prefix: str = LOG_PREFIX,
                 today: Callable[[], str] = lambda: time.strftime("%Y-%m-%d")) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.today = today
        self._lock = threading.Lock()
        self._file: Optional[BinaryIO] = None
        self._day: Optional[str] = None
        self._at_line_start = True
        self._disabled = False

    def path(self, day: str) -> Path:
        return self.directory / f"{self.prefix}-{day}.log"

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if self._disabled:
                return
            try:
                day = self.today()
                if self._file is not None and day != self._day and not self._at_line_start:
                    # Finish the line in progress in the old day's file, then
                    # rotate for the rest of the chunk. A read can end mid-line
                    # every time, so waiting for one that ends at a line
                    # boundary could keep yesterday's file open indefinitely.
                    end = data.find(b"\n") + 1
                    if not end:
                        self._append(data)
                        return
                    self._append(data[:end])
                    data = data[end:]
                    if not data:
                        return
                if self._file is None or day != self._day:
                    self._open(day)
                self._append(data)
            except OSError as exc:
                self._disable(exc)

    def _append(self, data: bytes) -> None:
        self._file.write(data)
        self._file.flush()
        self._at_line_start = data.endswith(b"\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _open(self, day: str) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path(day), "ab")
        self._day = day

    def _disable(self, exc: OSError) -> None:
        self._disabled = True
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        try:
            sys.stderr.write(f"[log_archive] archiving to {self.directory} disabled: {exc}\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


def _pump(source: BinaryIO, sink: BinaryIO, archive: DailyLog) -> None:
    """Copy *source* to *sink* and the archive as it arrives (no line buffering)."""
    fd = source.fileno()
    sink_ok = True
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        if sink_ok:
            try:
                sink.write(chunk)
                sink.flush()
            except (OSError, ValueError):
                sink_ok = False  # Keep archiving even if our own output went away.
        archive.write(chunk)


def run(command: Sequence[str], archive: DailyLog, *,
        stdout: Optional[BinaryIO] = None, stderr: Optional[BinaryIO] = None) -> int:
    """Run *command* with its output teed into *archive*; return its exit code."""
    posix = os.name == "posix"
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        # Isolate the command and its children from terminal signals sent to
        # the wrapper's group. The command coordinates its workers' shutdown.
        start_new_session=posix,
    )

    def forward(signum, _frame) -> None:
        try:
            # Signalling the whole group kills model/compile workers before
            # the gateway can stop them and unlink their named semaphores.
            process.send_signal(signum)
        except OSError:
            pass  # Already gone.

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)

    pumps = [
        threading.Thread(target=_pump, args=(process.stdout, stdout or sys.stdout.buffer, archive),
                         daemon=True),
        threading.Thread(target=_pump, args=(process.stderr, stderr or sys.stderr.buffer, archive),
                         daemon=True),
    ]
    try:
        for pump in pumps:
            pump.start()
        code = process.wait()
        for pump in pumps:
            # A grandchild may keep a pipe open after the command exits.
            pump.join(timeout=5)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        archive.close()
    # A command killed by a signal exits like a shell reports it.
    return code if code >= 0 else 128 - code


def main(argv: Optional[Sequence[str]] = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        sys.stderr.write("usage: python -m src.server.log_archive <command> [args...]\n")
        return 2
    return run(command, DailyLog(log_dir()))


if __name__ == "__main__":
    sys.exit(main())
