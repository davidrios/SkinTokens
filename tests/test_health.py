"""Health reports whether another generation request can be admitted."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

import serve


def get_health():
    async def request():
        # Skip the model-loading lifespan; fixtures supply the pool state.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=serve.app), base_url="http://testserver"
        ) as client:
            return await client.get("/health", headers={"Origin": "http://localhost:3000"})

    return asyncio.run(request())


@pytest.mark.parametrize("active_jobs", [0, 1])
def test_health_tracks_admission_capacity(monkeypatch, active_jobs):
    pool = SimpleNamespace(status=lambda: {"workers": 1, "queued": 0})
    serve.state.pool = serve.state.cpu_pool = pool
    serve.state.active_jobs = active_jobs
    monkeypatch.setattr(serve, "_admission_limit", lambda: 8)

    # Uploads also occupy admission slots, even before generation starts.
    for admitted, expected in [(7, "ok"), (8, "busy"), (9, "busy"), (7, "ok")]:
        serve.state.admitted = admitted
        response = get_health()
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
        body = response.json()
        assert body["status"] == expected
        assert body["admitted"] == admitted
        assert body["max_admitted"] == 8
        assert body["busy"] is (active_jobs > 0)


def test_health_loading_takes_precedence_over_full_queue():
    serve.state.admitted = 1000
    response = get_health()
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}
    assert response.headers["access-control-allow-origin"] == "*"
