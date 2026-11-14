"""Untrusted (uploaded) geometry: no server file access, bounded resources."""

import asyncio
import base64
import json
from pathlib import Path

import numpy as np
import pytest

import serve
from src.rig_package.glb_io import Builder, InputLimits
from src.rig_package.info.asset import Asset
from src.rig_package.parser.glb import GlbParser, transfer_rigging

LIMITS = InputLimits()
SECRET = b"SECRET-SERVER-FILE-CONTENT"
TRIANGLE = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4")


def _gltf(path: Path, *, buffer_uri=None, image_uri=None):
    data = TRIANGLE.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(data),
                     "uri": buffer_uri or "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": [{"buffer": 0, "byteLength": len(data)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    if image_uri:
        doc["images"] = [{"uri": image_uri, "mimeType": "image/png"}]
        doc["textures"] = [{"source": 0}]
        doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        doc["meshes"][0]["primitives"][0]["material"] = 0
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def secret(tmp_path):
    path = tmp_path / "secret.bin"
    path.write_bytes(SECRET)
    upload = tmp_path / "upload"
    upload.mkdir()
    return path, upload


def _rig(asset):
    return Asset.from_data(vertices=asset.vertices, joints=np.zeros((1, 3)), parents=np.array([-1]),
                           joint_names=["root"], skin=np.ones((asset.N, 1), np.float32))


def test_untrusted_gltf_cannot_read_external_buffers(secret):
    secret_path, upload = secret
    for uri in ("../secret.bin", str(secret_path), secret_path.as_uri()):
        path = _gltf(upload / "evil.gltf", buffer_uri=uri)
        with pytest.raises(ValueError):
            GlbParser.load(str(path), limits=LIMITS)


def test_untrusted_gltf_images_are_never_read_or_exported(secret, tmp_path):
    secret_path, upload = secret
    for uri in ("../secret.bin", str(secret_path)):
        path = _gltf(upload / "evil.gltf", image_uri=uri)
        asset = GlbParser.load(str(path), limits=LIMITS)
        out = tmp_path / "out.glb"
        transfer_rigging(_rig(asset), str(path), str(out), limits=LIMITS)
        assert SECRET not in out.read_bytes()


@pytest.mark.parametrize("limits, match", [
    (InputLimits(max_vertices=2), "vertex count"),
    (InputLimits(max_faces=0), "triangle count"),
    (InputLimits(max_mesh_instances=0), "mesh instance"),
    (InputLimits(max_primitives=0), "primitive count"),
    (InputLimits(max_file_bytes=10), "file size"),
])
def test_mesh_limits_checked_before_decoding(tmp_path, limits, match):
    path = _gltf(tmp_path / "tri.gltf")
    with pytest.raises(ValueError, match=match):
        GlbParser.load(str(path), limits=limits)


@pytest.mark.parametrize('mode', [0, 1, 2, 3])
@pytest.mark.parametrize('limits', [None, InputLimits(max_vertices=3, max_faces=1)])
def test_points_and_lines_are_skipped_before_decoding(tmp_path, mode, limits):
    path = _gltf(tmp_path / 'mixed.gltf')
    doc = json.loads(path.read_text())
    # These accessors must never be decoded or counted as triangle geometry.
    doc['accessors'].extend([
        {'componentType': 5126, 'count': 10**12, 'type': 'VEC3'},
        {'componentType': 5125, 'count': 10**12, 'type': 'SCALAR'},
    ])
    ignored = {'mode': mode, 'attributes': {'POSITION': 1}, 'indices': 2}
    doc['meshes'][0]['primitives'].insert(0, ignored)
    doc['meshes'].append({'primitives': [ignored]})
    doc['nodes'].append({'mesh': 1})
    doc['scenes'][0]['nodes'].append(1)
    path.write_text(json.dumps(doc))
    asset = GlbParser.load(str(path), limits=limits)
    assert (asset.N, asset.F, asset.P) == (3, 1, 1)
    np.testing.assert_array_equal(asset.faces, [[0, 1, 2]])
    out = tmp_path / 'out.glb'
    transfer_rigging(_rig(asset), str(path), str(out), limits=limits)
    reloaded = GlbParser.load(str(out))
    np.testing.assert_allclose(reloaded.vertices, asset.vertices, atol=1e-6)
    np.testing.assert_array_equal(reloaded.faces, asset.faces)

    doc['meshes'][0]['primitives'] = [ignored]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='No triangle geometry'):
        GlbParser.load(str(path), limits=limits)


@pytest.mark.parametrize('location', ['extensionsUsed', 'extensionsRequired', 'node'])
@pytest.mark.parametrize('limits', [None, LIMITS])
def test_gpu_instancing_is_rejected(tmp_path, location, limits):
    path = _gltf(tmp_path / 'instanced.gltf')
    doc = json.loads(path.read_text())
    extension = 'EXT_mesh_gpu_instancing'
    if location == 'node':
        doc['nodes'][0]['extensions'] = {extension: {'attributes': {'TRANSLATION': 0}}}
    else:
        doc[location] = [extension]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=extension):
        GlbParser.load(str(path), limits=limits)


def _skinned(tmp_path, joints, name="j"):
    """GLB with a triangle skinned to ``joints`` chained joints."""
    b = Builder()
    pos = b.accessor(TRIANGLE, "VEC3", bounds=True)
    ids = b.accessor(np.zeros((3, 4)), "VEC4", component=5123)
    w = b.accessor(np.tile([1., 0, 0, 0], (3, 1)), "VEC4")
    b.data["meshes"] = [{"primitives": [{"attributes": {"POSITION": pos, "JOINTS_0": ids, "WEIGHTS_0": w}}]}]
    b.data["nodes"] = [{"mesh": 0, "skin": 0}] + [
        {"name": f"{name}{i}", **({"children": [i + 2]} if i + 1 < joints else {})} for i in range(joints)]
    b.data["skins"] = [{"joints": list(range(1, joints + 1))}]
    b.data["scenes"] = [{"nodes": [0, 1]}]
    path = tmp_path / "skinned.glb"
    b.write(path)
    return path


def test_input_skin_and_joints_are_ignored(tmp_path):
    """A skinned upload is read as a plain mesh, however large its rig is."""
    asset = GlbParser.load(str(_skinned(tmp_path, 2000, name="x" * 5000)), limits=LIMITS)
    assert asset.N == 3
    assert asset.joint_names is None and asset.parents is None and asset.skin is None


def test_malformed_structure_is_value_error(tmp_path):
    path = _gltf(tmp_path / "bad.gltf")
    doc = json.loads(path.read_text())
    doc["accessors"][0].pop("componentType")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Malformed glTF"):
        GlbParser.load(str(path), limits=LIMITS)
    path.write_text('{"a":' * 100000 + "1" + "}" * 100000)
    with pytest.raises(ValueError):
        GlbParser.load(str(path), limits=LIMITS)


def test_limits_from_env(monkeypatch):
    monkeypatch.setenv("SKINTOKENS_INPUT_MAX_VERTICES", "123")
    monkeypatch.setenv("SKINTOKENS_MAX_UPLOAD_MB", "128")
    limits = InputLimits.from_env()
    assert limits.max_vertices == 123
    assert limits.max_file_bytes == 128 * 1024 * 1024


def test_serve_never_uses_client_filename_as_path(monkeypatch):
    """The gateway passes only the upload's suffix to the worker, which writes
    it to a fixed name in its own temp directory."""
    from src.server import pipeline
    from src.server.worker import Cancellation

    seen = []

    def fake_prepare(input_path, *args, **kwargs):
        seen.append(Path(input_path))
        raise RuntimeError("stop")

    monkeypatch.setitem(pipeline.STAGES, "prepare", fake_prepare)
    for name in ("../../escape.glb", "C:/Windows/escape.glb", "/etc/escape.glb", "a/../../b.glb"):
        suffix = serve._validate_upload(b"data", name)
        with pytest.raises(RuntimeError, match="stop"):
            pipeline.run_job({"suffix": suffix, "request_id": "req"}, b"data", Cancellation())
    assert len(seen) == 4
    assert all(p.name == "input.glb" and p.parent.name.startswith("skintokens_input_") for p in seen)
    assert not any(p.parent.exists() for p in seen)


def test_serve_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(serve, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="too large"):
        asyncio.run(serve._generate(b"12345", "model.glb", serve.GenParams(), "req"))
