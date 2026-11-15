"""The batching GPU worker's loop (src/server/pipeline.py ``serve``), with a
stand-in decoder and model, so it runs without a GPU.

Covered guarantees:
  * a request's point clouds decode as separate rows, up to the decoder's
    slots at once; the request is answered once all of them have ended, with
    its predictions in sample order;
  * a row that fails fails only its own request, and frees the request's
    other rows;
  * a cancelled request leaves the decoder and is answered with
    GenerationCancelled, while the others carry on;
  * progress reports follow the rows' decoding.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.server import pipeline
from src.server.stages import InferInput, pack, unpack
from src.server.worker import Cancellation, GenerationCancelled, Stopped


@pytest.mark.parametrize("slots", [0, 1, 2, 3, 4])
def test_batch_slots_environment_preserves_supported_values(monkeypatch, slots):
    monkeypatch.setenv("SKINTOKENS_BATCH_SLOTS", str(slots))
    assert pipeline.config._batch_slots() == slots


class FakeDecoder:
    """Rows end after as many steps as their prompt has positions."""

    def __init__(self, slots):
        self.slots = slots
        self.rows = {}  # key -> steps left
        self.peak = 0
        self.inserted = []

    @property
    def active(self):
        return len(self.rows)

    @property
    def free(self):
        return self.slots - len(self.rows)

    def insert(self, key, inputs_embeds, start_tokens):
        assert self.free
        self.rows[key] = inputs_embeds.shape[0]
        self.inserted.append(key)
        self.peak = max(self.peak, len(self.rows))

    def remove(self, key):
        return self.rows.pop(key, None) is not None

    def step(self):
        finished = []
        for key in list(self.rows):
            self.rows[key] -= 1
            if self.rows[key] <= 0:
                del self.rows[key]
                finished.append(SimpleNamespace(key=key, generated_ids=torch.tensor([1, 2])))
        return finished

    def progress(self, key):
        return {"generated": 1, "in_skin": 0, "switch_len": 0, "bones": 0} if key in self.rows else None


class FakeModel:
    """A point cloud's first coordinate is its row's length in steps; a
    negative one makes its result fail."""
    device = torch.device("cpu")
    tokens_per_skin = 2

    def prepare_prompts(self, vertices, normals, cls):
        return [{"inputs_embeds": torch.zeros(max(1, abs(int(v[0, 0]))), 4),
                 "start_tokens": [0, 1], "cond": v, "cond_latents": None,
                 "fail": bool(v[0, 0] < 0)} for v in vertices]

    def result_from_ids(self, start_tokens, generated_ids, cond, cond_latents, progress_callback=None):
        if cond[0, 0] < 0:
            raise ValueError("undecodable skeleton")
        return SimpleNamespace(length=int(cond[0, 0]))

    def attach_asset(self, res, mesh, sampled_vertices, interpolate_skin=True):
        res.asset = SimpleNamespace(mesh=mesh, length=res.length)
        return res


class FakeJob:
    def __init__(self, job_id, lengths, cancel_after=None):
        self.id = job_id
        vertices = np.zeros((len(lengths), 3, 3), np.float32)
        vertices[:, 0, 0] = lengths
        # num_samples is left out: SKINTOKENS_MAX_NUM_SAMPLES caps it, and the
        # rows come from the point clouds.
        self.params = {"seed": 7, "output": "rig", "request_id": f"r{job_id}"}
        self.upload = pack(InferInput(vertices=vertices, normals=vertices.copy(),
                                      cls="articulation", mesh=f"mesh{job_id}"))
        self.cancellation = Cancellation()
        self.cancel_after = cancel_after
        self.events, self.result, self.error = [], None, None

    def progress(self, event):
        self.events.append(event)

    def reply(self, value):
        self.result = unpack(value)

    def fail(self, exc):
        self.error = exc


class FakeSource:
    """Hands out the jobs one per call; cancels a job after its given number
    of calls; stops once nothing is left to do."""

    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.all = list(jobs)
        self.calls = 0

    def next(self, block):
        self.calls += 1
        for job in self.all:
            if job.cancel_after is not None and self.calls >= job.cancel_after:
                job.cancellation.cancel("client requested cancellation")
        if self.jobs:
            return self.jobs.pop(0)
        if block:
            raise Stopped()
        return None


@pytest.fixture
def stand_ins(monkeypatch):
    def install(slots):
        decoder = FakeDecoder(slots)
        monkeypatch.setattr(pipeline.state, "decoder", decoder)
        monkeypatch.setattr(pipeline.state, "model", FakeModel())
        monkeypatch.setattr(pipeline, "finish_skin", lambda asset, *a, **k: f"skin{asset.length}")
        monkeypatch.setattr(pipeline, "PROGRESS_EVERY_STEPS", 1)
        return decoder
    return install


def run(jobs):
    with pytest.raises(Stopped):
        pipeline.serve(FakeSource(jobs))


def test_rows_share_the_slots_and_each_request_gets_its_samples_in_order(stand_ins):
    decoder = stand_ins(slots=2)
    jobs = [FakeJob(1, [5, 2, 9]), FakeJob(2, [3])]
    run(jobs)
    assert decoder.peak == 2 and not decoder.rows
    for job, lengths in zip(jobs, ([5, 2, 9], [3])):
        assert job.error is None
        assert [p.asset.length for p in job.result] == lengths
        assert [p.skin for p in job.result] == [f"skin{n}" for n in lengths]
        assert job.result[0].asset.mesh == f"mesh{job.id}"
        steps = [e["step"] for e in job.events]
        assert steps[:2] == ["running inference", "model sampling"]
        assert steps[-1] == "inference done"
        percents = [e["progress"] for e in job.events]
        assert percents == sorted(percents)


def test_a_failing_row_fails_only_its_request(stand_ins):
    decoder = stand_ins(slots=4)
    bad, good = FakeJob(1, [2, -1, 30]), FakeJob(2, [4])
    run([bad, good])
    assert isinstance(bad.error, ValueError) and bad.result is None
    assert (1, 2) not in decoder.rows  # its long row was freed, not run out
    assert good.error is None and [p.asset.length for p in good.result] == [4]


def test_a_cancelled_request_leaves_the_decoder(stand_ins):
    decoder = stand_ins(slots=2)
    cancelled, other = FakeJob(1, [500], cancel_after=6), FakeJob(2, [20])
    run([cancelled, other])
    assert isinstance(cancelled.error, GenerationCancelled)
    assert "client requested" in str(cancelled.error)
    assert other.error is None and other.result is not None
    assert not decoder.rows


def test_a_request_without_a_seed_is_refused(stand_ins):
    stand_ins(slots=1)
    job = FakeJob(1, [2])
    job.params["seed"] = None
    run([job])
    assert isinstance(job.error, ValueError) and job.result is None
