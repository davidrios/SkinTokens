"""Logging: one summary line per request, daily archive, failure dumps."""
import logging

import serve
from src.server import log_archive, logs


def test_daily_log_rotates_at_line_boundary(tmp_path):
    day = ["2026-01-01"]
    archive = log_archive.DailyLog(tmp_path, today=lambda: day[0])
    archive.write(b"one\npart")
    day[0] = "2026-01-02"
    archive.write(b"ial\ntwo\n")
    archive.close()
    assert (tmp_path / "skintokens-2026-01-01.log").read_bytes() == b"one\npartial\n"
    assert (tmp_path / "skintokens-2026-01-02.log").read_bytes() == b"two\n"


def test_failed_request_is_dumped_with_one_log_line(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SKINTOKENS_OUTPUT_ROOT", str(tmp_path))
    record = serve._RequestRecord("abc123", ("1.2.3.4", 5))
    record.raw_header = '{"filename": "a.glb"}'
    record.header_valid = True
    record.filename = "a.glb"
    record.upload = b"glTFdata"
    record.upload_bytes = 8
    serve.logger.setLevel(logging.DEBUG)
    serve.logger.debug("stage detail")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record.fail(exc)
    serve.logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="skintokens.serve"):
        record.save()
    serve.logger.propagate = False
    lines = [r for r in caplog.records if r.getMessage().startswith("[abc123]")]
    assert len(lines) == 1 and lines[0].levelno == logging.ERROR
    assert "status=error" in lines[0].getMessage() and "boom" in lines[0].getMessage()
    dump = next((tmp_path / "dump").glob("*/abc123"))
    assert (dump / "input.glb").read_bytes() == b"glTFdata"
    assert (dump / "params.json").exists()
    assert "RuntimeError: boom" in (dump / "traceback.txt").read_text()
    assert "stage detail" in (dump / "trace.log").read_text()


def test_successful_request_writes_no_dump(tmp_path, monkeypatch):
    monkeypatch.setenv("SKINTOKENS_OUTPUT_ROOT", str(tmp_path))
    record = serve._RequestRecord("ok1", None)
    record.status = "done"
    record.save()
    assert not (tmp_path / "dump").exists()


def test_rejections_and_client_errors_write_no_dump(tmp_path, monkeypatch):
    monkeypatch.setenv("SKINTOKENS_OUTPUT_ROOT", str(tmp_path))
    busy = serve._RequestRecord("busy1", None)
    busy.reject("server busy: 18 requests running or queued; retry later")
    busy.save()
    assert busy.status == "rejected"
    bad = serve._RequestRecord("bad1", None)
    bad.raw_header = "not json"
    bad.fail(ValueError("invalid JSON message"))
    bad.save()
    assert bad.status == "error" and not bad.server_error
    assert not (tmp_path / "dump").exists()


def test_cancellation_and_benign_disconnect_write_no_dump(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("SKINTOKENS_OUTPUT_ROOT", str(tmp_path))
    cancelled = serve._RequestRecord("c1", None)
    cancelled.fail(asyncio.CancelledError())
    cancelled.save()
    assert cancelled.status == "cancelled" and not cancelled.server_error
    disconnected = serve._RequestRecord("d1", None)
    disconnected.fail(RuntimeError("cannot call 'send' once 'close' has already been called"))
    disconnected.save()
    assert disconnected.status == "disconnected" and not disconnected.server_error
    assert not (tmp_path / "dump").exists()


def test_generation_cancelled_is_cancelled_not_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SKINTOKENS_OUTPUT_ROOT", str(tmp_path))
    record = serve._RequestRecord("g1", None)
    record.fail(serve.GenerationCancelled("client requested cancellation"))
    record.save()
    assert record.status == "cancelled" and not record.server_error
    assert record.traceback is None
    assert not (tmp_path / "dump").exists()


def test_request_id_is_hhmmss_plus_16_hex():
    import re
    rid = serve.new_request_id()
    assert re.fullmatch(r"\d{6}-[0-9a-f]{16}", rid)
