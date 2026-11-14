"""Exercise container-style signals through the real log wrapper and pool."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


GATEWAY = '''
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from src.server.pool import WorkerPool

async def main():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    pool = WorkerPool("pool_fixtures:RESOURCES", 1)
    await pool.start()
    try:
        job = None
        if sys.argv[2] == "busy":
            started = asyncio.Event()
            async def on_start():
                started.set()
            job = asyncio.create_task(pool.submit(
                {"action": "sleep", "seconds": 0.5}, b"", on_start=on_start))
            await started.wait()
        Path(sys.argv[1]).write_text(json.dumps(
            [os.getpid(), pool._workers[0].process.pid]))
        await stop.wait()
    finally:
        await pool.close(timeout=5)
    if job is not None:
        await job
    assert all(w.process.exitcode == 0 for w in pool._workers)
    assert not pool._respawns
    print("clean shutdown", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
'''


@pytest.mark.skipif(os.name != "posix", reason="container signal semantics are POSIX")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("state", ["idle", "busy"])
def test_wrapper_allows_workers_to_release_resources(tmp_path, signum, state):
    import json

    root = Path(__file__).resolve().parents[1]
    gateway = tmp_path / "gateway.py"
    gateway.write_text(GATEWAY)
    ready = tmp_path / "ready.json"
    env = {**os.environ, "SKINTOKENS_OUTPUT_ROOT": str(tmp_path / "outputs"),
           "PYTHONPATH": os.pathsep.join([str(root), str(root / "tests")])}
    process = subprocess.Popen(
        [sys.executable, "-m", "src.server.log_archive", sys.executable,
         str(gateway), str(ready), state],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("gateway did not start")
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=5)
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        assert b"clean shutdown" in stdout
        assert b"resource_tracker" not in stderr
        archived = b"".join(p.read_bytes() for p in (tmp_path / "outputs/logs").glob("*.log"))
        assert b"clean shutdown" in archived
        assert b"resource_tracker" not in archived
    finally:
        # Also clean up descendants if a regression prevents normal shutdown.
        if ready.exists():
            for pid in json.loads(ready.read_text()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
