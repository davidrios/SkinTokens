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
    """The gateway passes only the upload's suffix to the CPU worker, which
    keeps the upload in memory under a fixed name (nothing is written to disk)."""
    from src.server import stages
    from src.server.worker import Cancellation

    seen = []

    def fake_prepare(source, *args, **kwargs):
        seen.append(source)
        raise RuntimeError("stop")

    monkeypatch.setitem(stages.STAGES, "prepare", fake_prepare)
    for name in ("../../escape.glb", "C:/Windows/escape.glb", "/etc/escape.glb", "a/../../b.glb"):
        suffix = serve._validate_upload(b"data", name)
        with pytest.raises(RuntimeError, match="stop"):
            stages.run_job({"stage": "prepare", "suffix": suffix, "request_id": "req", "seed": 1},
                           b"data", Cancellation())
    assert len(seen) == 4
    assert all(isinstance(s, stages.Upload) and s.name == "input.glb" and s.data == b"data"
               for s in seen)


def test_in_memory_upload_parses_and_is_size_limited(tmp_path):
    from src.rig_package.glb_io import load_scene
    b = Builder()
    pos = b.accessor(TRIANGLE, 'VEC3', bounds=True)
    b.data['meshes'] = [{'primitives': [{'attributes': {'POSITION': pos}}]}]
    b.data['nodes'] = [{'mesh': 0}]
    b.data['scenes'] = [{'nodes': [0]}]
    data = b.to_bytes()
    path = tmp_path / 'input.glb'
    path.write_bytes(data)
    memory_scene = load_scene(data, name='input.glb')
    file_scene = load_scene(path)
    assert memory_scene.vertex_count == file_scene.vertex_count == 3
    np.testing.assert_array_equal(memory_scene.vertices(), TRIANGLE)
    np.testing.assert_array_equal(memory_scene.vertices(), file_scene.vertices())
    with pytest.raises(ValueError, match="file size"):
        load_scene(data, limits=InputLimits(max_file_bytes=len(data) - 1), name="input.glb")
    with pytest.raises(ValueError, match="unsupported type"):
        load_scene(data, name="input.obj")


def test_serve_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(serve, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="too large"):
        asyncio.run(serve._generate(b"12345", "model.glb", serve.GenParams(), "req"))


def _gltf_doc(path: Path) -> dict:
    return json.loads(_gltf(path).read_text())


def _load_doc(path: Path, doc: dict, limits=LIMITS):
    path.write_text(json.dumps(doc))
    return GlbParser.load(str(path), limits=limits)


def test_negative_counts_cannot_cancel_out_the_limits(tmp_path):
    """A negative count used to subtract from the summed vertex count, so a
    huge zero-filled accessor passed the limit (GBs from a 1 KB upload)."""
    path = tmp_path / "neg.gltf"
    doc = _gltf_doc(path)
    big = 2**26 // 3
    doc["accessors"] += [{"componentType": 5126, "count": big, "type": "VEC3"},
                         {"componentType": 5126, "count": -big, "type": "VEC3"}]
    doc["meshes"][0]["primitives"] = [{"attributes": {"POSITION": 1}}, {"attributes": {"POSITION": 2}}]
    with pytest.raises(ValueError, match="Invalid glTF accessor count"):
        _load_doc(path, doc)
    for count in (-3, "3", True, 3.5):
        doc = _gltf_doc(path)
        doc["accessors"][0]["count"] = count
        with pytest.raises(ValueError):
            _load_doc(path, doc)
    doc = _gltf_doc(path)
    doc["accessors"][0]["count"] = 3.0  # an integral JSON number is a count
    assert _load_doc(path, doc).N == 3


def test_sparse_indices_out_of_range_rejected(tmp_path):
    path = tmp_path / "sparse.gltf"
    doc = _gltf_doc(path)
    data = np.array([5], "<u4").tobytes() + TRIANGLE[:1].tobytes()
    doc["buffers"].append({"byteLength": len(data),
                           "uri": "data:;base64," + base64.b64encode(data).decode()})
    doc["bufferViews"] += [{"buffer": 1, "byteLength": 4},
                           {"buffer": 1, "byteOffset": 4, "byteLength": 12}]
    doc["accessors"][0]["sparse"] = {"count": 1, "indices": {"bufferView": 1, "componentType": 5125},
                                     "values": {"bufferView": 2}}
    with pytest.raises(ValueError, match="Sparse accessor index out of range"):
        _load_doc(path, doc)


def test_node_count_limited_before_the_hierarchy_walk(tmp_path):
    """Empty nodes cost no vertices but ~80 us each to walk: 1M of them (a
    4 MB upload) took 80 s per request before the node limit."""
    path = tmp_path / "nodes.gltf"
    doc = _gltf_doc(path)
    doc["nodes"] += [{}] * 10
    doc["scenes"][0]["nodes"] = list(range(11))
    assert _load_doc(path, doc, InputLimits(max_nodes=11)).N == 3
    with pytest.raises(ValueError, match="node count"):
        _load_doc(path, doc, InputLimits(max_nodes=10))


@pytest.mark.parametrize("roots", [[0, -1], [5], [0.0], [True]])
def test_invalid_scene_roots_rejected(tmp_path, roots):
    path = tmp_path / "roots.gltf"
    doc = _gltf_doc(path)
    doc["scenes"][0]["nodes"] = roots
    with pytest.raises(ValueError, match="scene root"):
        _load_doc(path, doc)


@pytest.mark.parametrize("vertices, node, match", [
    ([[np.nan] * 3, [1, 0, 0], [0, 1, 0]], {}, "non-finite"),
    ([[np.inf, 0, 0], [1, 0, 0], [0, 1, 0]], {}, "non-finite"),
    ([[3e38, 0, 0], [-3e38, 0, 0], [0, 1, 0]], {}, "non-finite"),  # extent overflows float32
    (TRIANGLE, {"matrix": [float("nan")] * 16}, "non-finite"),
    (TRIANGLE, {"scale": [1e308] * 3}, "non-finite"),
    (np.zeros((3, 3)), {}, "zero extent"),
    (TRIANGLE, {"scale": [0, 0, 0]}, "zero extent"),
])
@pytest.mark.parametrize("limits", [None, LIMITS])
def test_unusable_geometry_rejected(tmp_path, vertices, node, match, limits):
    """Such geometry used to reach the model as NaN (normalization divides by the extent)."""
    path = tmp_path / "geom.gltf"
    doc = _gltf_doc(path)
    data = np.asarray(vertices, "<f4").tobytes()
    doc["buffers"][0]["uri"] = "data:;base64," + base64.b64encode(data).decode()
    doc["nodes"][0].update(node)
    with pytest.raises(ValueError, match=match):
        _load_doc(path, doc, limits)


def test_export_checks_normal_count_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "normals.gltf"
    doc = _gltf_doc(path)
    doc["accessors"].append({"componentType": 5126, "count": 2**26 // 3, "type": "VEC3"})
    doc["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] = 1
    asset = _load_doc(path, doc)
    from src.rig_package.glb_io.document import Document
    decoded = []
    original = Document.accessor
    monkeypatch.setattr(Document, "accessor", lambda self, i: decoded.append(i) or original(self, i))
    with pytest.raises(ValueError, match="NORMAL count"):
        transfer_rigging(_rig(asset), str(path), str(tmp_path / "out.glb"), limits=LIMITS)
    assert 1 not in decoded


# --------------------------------------------------------------------------- #
# WebSocket gateway
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, match", [
    ("[1, 2]", "JSON object"),
    ('"str"', "JSON object"),
    ("{not json", "invalid JSON"),
    ('{"a":' * 100000 + "1" + "}" * 100000, "too large"),
    ('{"x": "' + "a" * 70000 + '"}', "too large"),
], ids=["list", "string", "invalid", "deep", "long"])
def test_text_messages_bounded_and_validated(text, match):
    with pytest.raises(ValueError, match=match):
        serve._parse_text_message(text)
    with pytest.raises(ValueError):
        serve._parse_text_message(b"{}")
    assert serve._parse_text_message('{"type": "cancel"}') == {"type": "cancel"}


@pytest.mark.parametrize("filename", [{"a": 1}, 5, None, "a" * 256 + ".glb"])
def test_request_filename_validated(filename):
    with pytest.raises(ValueError, match="filename"):
        serve._request_filename({"filename": filename})
    assert serve._request_filename({}) == "input.glb"


def test_internal_error_details_stay_in_the_log():
    assert serve._client_error_message(ValueError("bad mesh"), "r1") == "bad mesh"
    message = serve._client_error_message(RuntimeError(r"C:\srv\secret\path.py failed"), "r1")
    assert "secret" not in message and "RuntimeError" in message and "r1" in message


class _FakeWs:
    """Just enough of a Starlette WebSocket for the handler's rejection paths."""

    def __init__(self, *messages, hang=False):
        self.incoming = list(messages)
        self.hang = hang
        self.sent = []
        self.closed = False
        self.client = ("test", 0)

    async def accept(self):
        pass

    async def _next(self, kind):
        if not self.incoming:
            if self.hang:
                await asyncio.sleep(3600)
            raise serve.WebSocketDisconnect()
        if not isinstance(self.incoming[0], kind):
            raise KeyError("bytes" if kind is bytes else "text")
        return self.incoming.pop(0)

    async def receive_text(self):
        return await self._next(str)

    async def receive_bytes(self):
        return await self._next(bytes)

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self):
        self.closed = True


def _run_ws(ws):
    serve.state.pool = object()  # not used on the paths under test
    serve.state.cpu_pool = object()
    asyncio.run(serve.ws_generate(ws))
    return ws.sent[-1]["message"] if ws.sent else None


def test_ws_idle_client_times_out(monkeypatch):
    monkeypatch.setattr(serve.config, "HEADER_TIMEOUT_SEC", 0.05)
    assert "no request header" in _run_ws(_FakeWs(hang=True))
    monkeypatch.setattr(serve.config, "UPLOAD_TIMEOUT_SEC", 0.05)
    ws = _FakeWs('{"filename": "a.glb"}', hang=True)
    assert "not received" in _run_ws(ws)
    assert serve.state.admitted == 0


def test_ws_rejects_bad_frames_and_params():
    assert "JSON request header" in _run_ws(_FakeWs(b"binary first"))
    assert "binary message" in _run_ws(_FakeWs('{"filename": "a.glb"}', "text file"))
    assert "JSON object" in _run_ws(_FakeWs("[1, 2, 3]"))
    # Parameters are validated before the upload is read.
    ws = _FakeWs('{"num_samples": 9999}', b"never read")
    assert "num_samples" in _run_ws(ws)
    assert ws.incoming == [b"never read"]
    assert serve.state.admitted == 0


def test_ws_refuses_requests_beyond_the_queue_limit(monkeypatch):
    monkeypatch.setattr(serve.config, "WORKERS_PER_GPU", 1)
    monkeypatch.setattr(serve.config, "CPU_WORKERS", 2)
    monkeypatch.setattr(serve.config, "MAX_QUEUE", 2)
    monkeypatch.setattr(serve, "visible_gpu_count", lambda: 1)
    # 1 GPU worker + 2 CPU workers + 2 queued.
    serve.state.admitted = 5
    ws = _FakeWs('{"filename": "a.glb"}', b"upload")
    assert "server busy" in _run_ws(ws)
    assert ws.incoming == [b"upload"]  # refused before receiving the upload
    assert serve.state.admitted == 5
