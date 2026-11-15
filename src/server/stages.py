"""The CPU stages of a request, as the CPU worker processes run them.

A request runs in three stages on two pools of worker processes
(:mod:`src.server.pool`), so that while one request's rig is being sampled on
the GPU, the CPU stages of other requests run beside it:

1. ``prepare`` (here, a CPU worker): parse the upload, transform it and sample
   the point clouds the model is conditioned on.
2. ``infer`` (:mod:`src.server.pipeline`, a GPU worker): sample the rig and
   interpolate its skin onto the mesh vertices.
3. ``export`` (here, a CPU worker): transfer the rig onto the parsed file and
   encode the rig payload or GLB.

What a stage hands to the next crosses the gateway as pickled bytes (see
:func:`pack`), which the gateway forwards without unpickling.

This module stays free of model imports (no ``src.model``, no CUDA): CPU
workers start with no GPU visible. Every stage runs under seeds derived from
the request's ``seed`` (:mod:`src.data.seeding`), so which process runs it
does not change the result.
"""
from __future__ import annotations

import dataclasses
import io
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

from src.server.config import GenParams
from src.server.worker import Cancellation, ServiceSpec

logger = logging.getLogger("skintokens.serve")

GROUP_PER_VERTEX = 4  # influences per vertex in the exported GLB


def pack(value: Any) -> bytes:
    """A stage's hand-off to the next stage, as the gateway forwards it."""
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpack(data: bytes) -> Any:
    """Only ever applied to :func:`pack` output of this server's own workers."""
    return pickle.loads(data)


class _State:
    transform = None


state = _State()


@dataclasses.dataclass(frozen=True)
class Upload:
    """An uploaded file, kept in memory (never written to disk). ``name`` is
    chosen by the server (``input`` + the validated suffix); the client's
    filename is used for nothing."""
    data: bytes
    name: str


@dataclasses.dataclass
class InferInput:
    """What the GPU stage needs from ``prepare``: the point clouds (numpy,
    ``(num_samples, P, 3)``) and the mesh the predicted rig is attached to."""
    vertices: Any
    normals: Any
    cls: str
    mesh: Any  # an Asset with only vertices, faces, cls and path


@dataclasses.dataclass
class Prediction:
    """One sample's predicted rig: the asset (mesh + skeleton) and its skin,
    already interpolated onto the mesh vertices (a ``SparseSkin``)."""
    asset: Any
    skin: Any


def _load_input(source, limits):
    """Parse ``source`` (an :class:`Upload` or a file path) as a scene."""
    from src.rig_package.glb_io import load_scene
    if isinstance(source, Upload):
        return load_scene(source.data, limits=limits, name=source.name)
    return load_scene(source, limits=limits)


def prepare_inputs(source, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback=None):
    """Stage 1 (CPU): parse the input (an :class:`Upload`, or a file path) into
    a single model batch holding ``params.num_samples`` point clouds, cloud
    ``i`` sampled under ``derive_seed(params.seed, i)``.

    Returns ``(InferInput, scene)``: the parsed scene stays with the CPU side,
    so export need not parse the file again.
    """
    from src.data.seeding import sample_point_clouds, seeded_rng
    from src.rig_package.glb_io import InputLimits
    from src.rig_package.info.asset import Asset
    from src.rig_package.parser.glb import GlbParser

    cancellation.raise_if_cancelled()
    if progress_callback:
        progress_callback(5, "building dataset")

    transform = state.transform
    scene = _load_input(source, InputLimits.from_env())
    # The model only sees normals at its sampled points (the export copies
    # the file's own). When the transform moves the vertices, normals are
    # not loaded and the sampler computes them at its points only.
    asset = GlbParser.asset_from_scene(scene, include_normals=not transform.rebuilds_normals)
    # The class the model is conditioned on (the dataset key the dataloader
    # path used to derive it from).
    asset.cls = "articulation"
    asset.path = source.name if isinstance(source, Upload) else str(source)
    cancellation.raise_if_cancelled()

    with seeded_rng(params.seed):
        transform.apply(asset=asset, sample=False)
    cancellation.raise_if_cancelled()

    # One cloud per sample (the transform's own single cloud is not drawn).
    vertices, normals = sample_point_clouds(asset, transform.sampler,
                                            params.seed, params.num_samples)
    # The predicted asset copies only these fields (TokenRig.predict_step).
    mesh = Asset.from_data(vertices=asset.vertices, faces=asset.faces,
                           cls=asset.cls, path=asset.path)
    infer_input = InferInput(vertices=vertices.numpy(), normals=normals.numpy(),
                             cls=asset.cls, mesh=mesh)
    return infer_input, scene


def finish_skin(asset, request_id: str, sample_idx: int, device: Optional[str] = None):
    """Collapse the predicted skeleton's near-parent joints and interpolate
    the skin, still at the sampled points, onto the mesh vertices.

    Returns a ``SparseSkin`` (each vertex's ``GROUP_PER_VERTEX`` heaviest
    influences), or ``None`` when the asset carries a dense skin already.
    ``device`` is a CUDA device to interpolate on, else CPU threads.
    """
    from src.rig_package.glb_io import SparseSkin
    from src.rig_package.skin_interp import interpolate_top_k

    # Merging a collapsed joint's skin into its ancestor commutes with the
    # (linear) interpolation onto the mesh, so with the skin still at the
    # sampled points, it is merged there, on far fewer rows.
    collapsed_joints = asset.collapse_near_parent_joints()
    if collapsed_joints:
        logger.debug("[%s] sample=%d collapsed near-parent skeleton joints: %s",
                     request_id, sample_idx, ", ".join(collapsed_joints))
    sampled_skin = (asset.sampled_vertex_groups or {}).get("skin")
    if asset.skin is not None or sampled_skin is None:
        return None
    ids, weights = interpolate_top_k(
        asset.vertices, asset.sampled_vertices, sampled_skin,
        top_k=GROUP_PER_VERTEX, device=device)
    return SparseSkin(ids, weights)


def _report_sample_progress(progress_callback, sample_count: int, sample_idx: int,
                            phase: str) -> None:
    """Map a per-sample export phase to a percentage in the 85–99 band."""
    if not progress_callback:
        return
    sample_progress_start = 85
    sample_progress_span = 14
    base = sample_progress_start + (sample_progress_span * sample_idx) // sample_count
    done = sample_progress_start + (sample_progress_span * (sample_idx + 1)) // sample_count
    mid = base + max(1, (done - base) // 2)
    sample_label = f"sample {sample_idx + 1}/{sample_count}"
    if phase == "start":
        if sample_idx > 0:
            return
        step = f"exporting {sample_label}"
        percent = base
    elif phase == "exported":
        step = f"exported {sample_label}"
        percent = min(mid, done - 1)
    elif phase == "complete":
        step = f"finished {sample_label}"
        percent = done
    else:
        raise ValueError(f"unknown sample progress phase: {phase}")
    progress_callback(percent, step)


def export_samples(preds: list, params: GenParams, request_id: str,
                   cancellation: Cancellation, progress_callback,
                   tmp_output_dir: Optional[Path] = None, scene=None,
                   source=None) -> list[bytes]:
    """Stage 3 (CPU): export each predicted asset in the ``params.output``
    format, a rig payload or a GLB (sample ``i`` under
    ``derive_seed(params.seed, i)``: the rig transfer subsamples points).
    Both come from the same :class:`SkinnedRig`, so the GLB a client rebuilds
    from a rig payload is the GLB this would have returned.

    Each of ``preds`` has an ``asset`` and, optionally, a ``skin`` (a
    :class:`Prediction` from the GPU stage, whose skin is already on the mesh
    vertices); without one, the skin is finished here on CPU threads
    (:func:`finish_skin`).

    ``scene`` is the input already parsed by :func:`prepare_inputs`; without it
    ``source`` (an :class:`Upload` or a path, else the asset's own path) is
    parsed once here. ``tmp_output_dir`` is unused (results are built in memory)
    and kept for callers.
    """
    from src.data.seeding import derive_seed, seeded_rng
    from src.rig_package.glb_io import InputLimits, encode_rig_payload, write_skinned_glb
    from src.rig_package.parser.glb import transfer_rig

    sample_count = max(len(preds), 1)
    outputs: list[bytes] = []
    if scene is None and preds:
        scene = _load_input(source or preds[0].asset.path, InputLimits.from_env())

    for sample_idx, pred in enumerate(preds):
        cancellation.raise_if_cancelled()
        _report_sample_progress(progress_callback, sample_count, sample_idx, "start")

        asset = pred.asset
        assert asset is not None
        skin = getattr(pred, "skin", None)
        if skin is None:
            skin = finish_skin(asset, request_id, sample_idx)

        with seeded_rng(derive_seed(params.seed, sample_idx)):
            rig = transfer_rig(
                source_asset=asset,
                scene=scene,
                skin=skin,
                group_per_vertex=GROUP_PER_VERTEX,
                # Shift the whole scene so its bottom sits at Z=0.
                auto_ground=True,
            )
        if params.output == "rig":
            outputs.append(encode_rig_payload(scene, rig))
        else:
            out = io.BytesIO()  # the GLB is built in memory
            write_skinned_glb(scene, out, rig)
            outputs.append(out.getvalue())
        del rig
        cancellation.raise_if_cancelled()
        _report_sample_progress(progress_callback, sample_count, sample_idx, "exported")

    return outputs


class ProgressReporter:
    """Per-stage progress/timing; checks for cancellation at every report."""

    LOG_SKIPPED_PREFIXES = ("model sampling", "decoding joints")

    def __init__(self, request_id: str, cancellation: Cancellation, send=None):
        self.request_id = request_id
        self.cancellation = cancellation
        self.send = send
        self.started_at = time.monotonic()
        self.last_logged_at = self.started_at

    def report(self, percent: int, stage: str) -> None:
        self.cancellation.raise_if_cancelled()
        now = time.monotonic()
        elapsed = round(now - self.started_at, 2)
        if not stage.startswith(self.LOG_SKIPPED_PREFIXES):
            logger.debug("[%s] progress=%d%% stage=%s elapsed=%.2fs delta=%.2fs",
                        self.request_id, percent, stage, elapsed, now - self.last_logged_at)
            self.last_logged_at = now
        if self.send is not None:
            # elapsed_sec is this stage's; the gateway reports the request's own.
            self.send({"progress": percent, "step": stage, "elapsed_sec": elapsed})
        self.cancellation.raise_if_cancelled()


# Stage functions the CPU worker calls; tests replace them.
STAGES = {"prepare": prepare_inputs, "export": export_samples}


# --------------------------------------------------------------------------- #
# CPU worker service
# --------------------------------------------------------------------------- #
def load(options: Optional[dict] = None) -> dict:
    """Set up a CPU worker (its ``load``): the model's input transform, from
    the config a GPU worker reported, and no model."""
    import torch
    from src.data.transform import Transform

    options = options or {}
    threads = options.get("cpu_threads")
    if threads:
        torch.set_num_threads(int(threads))
    state.transform = Transform.parse(**options["predict_transform"])
    return {"stages": sorted(STAGES)}


def run_job(params: dict, payload: Any, cancellation: Cancellation, progress=None):
    """One CPU stage of a request (a CPU worker's ``run``).

    *params* holds the :class:`GenParams` fields plus ``stage``,
    ``request_id`` and, for ``prepare``, ``suffix`` (the upload's validated
    file extension).

    ``prepare``: *payload* is the upload; returns ``{"infer": bytes,
    "export": bytes}``, the packed hand-offs to the GPU and export stages.
    ``export``: *payload* is ``(export hand-off, packed predictions)``;
    returns one rig payload or GLB per sample.
    """
    params = dict(params)
    stage = params.pop("stage")
    request_id = params.pop("request_id", "-")
    suffix = params.pop("suffix", None)
    gen = GenParams(**params)
    if gen.seed is None:
        raise ValueError("the gateway sends every stage a seed")
    reporter = ProgressReporter(request_id, cancellation, progress)

    if stage == "prepare":
        # Parsed straight from memory. The client's filename is used for
        # nothing: only its (validated) suffix, which tells GLB from glTF.
        source = Upload(payload, f"input{suffix}")
        del payload
        infer_input, scene = STAGES["prepare"](source, gen, request_id, cancellation,
                                               reporter.report)
        return {"infer": pack(infer_input), "export": pack(scene)}
    if stage == "export":
        scene_data, preds_data = payload
        del payload
        outputs = STAGES["export"](unpack(preds_data), gen, request_id, cancellation,
                                   reporter.report, scene=unpack(scene_data))
        reporter.report(100, "complete")
        return outputs
    raise ValueError(f"unknown CPU stage {stage!r}")


SERVICE = ServiceSpec("skintokens-cpu", load, run_job)
