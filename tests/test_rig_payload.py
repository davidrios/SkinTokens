"""Rig payloads: the server's compact result, rebuilt into the GLB by the client."""
import asyncio
from pathlib import Path
import json
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

import serve
from src.rig_package.glb_io import (Builder, SparseSkin, build_glb, decode_rig_payload,
                                    encode_rig_payload, load_scene, scene_layout)
from src.rig_package.glb_io.rig_payload import MAGIC
from src.rig_package.info.asset import Asset
from src.rig_package.parser.glb import GlbParser, transfer_rig, transfer_rigging
from test_input_hardening import _FakeWs, _run_ws, _skinned


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _write(path, meshes, nodes, roots, builder):
    builder.data["meshes"] = meshes
    builder.data["nodes"] = nodes
    builder.data["scenes"] = [{"nodes": roots}]
    builder.write(path)
    return path


def _instances(path):
    """One mesh under a mirrored, scaled parent twice, plus a points primitive
    and a second triangle primitive: exercises the vertex-order contract."""
    b = Builder()
    tri = b.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), "VEC3", bounds=True)
    quad = b.accessor(np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float), "VEC3", bounds=True)
    nrm = b.accessor(np.array([[0, 0, 1]] * 3, float), "VEC3")
    idx = b.accessor(np.array([0, 1, 2, 0, 2, 3]), "SCALAR", component=5125)
    meshes = [{"name": "body", "primitives": [
        {"attributes": {"POSITION": tri, "NORMAL": nrm}},
        {"attributes": {"POSITION": tri}, "mode": 0},  # points: not part of the layout
        {"attributes": {"POSITION": quad}, "indices": idx},
    ]}]
    nodes = [{"name": "parent", "translation": [0, 2, 0], "scale": [-2, 2, 2], "children": [1, 2]},
             {"name": "a", "mesh": 0}, {"name": "b", "mesh": 0, "translation": [0, 0, 5]}]
    return _write(path, meshes, nodes, [0], b)


def _grid(path, n):
    """A single ``n`` x ``n`` grid (more than 65535 vertices for n > 256)."""
    b = Builder()
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 2, n))
    pos = np.stack([u.ravel(), v.ravel(), np.sin(3 * u.ravel()) * 0.2], axis=1)
    i = np.arange(n * n).reshape(n, n)[:-1, :-1].ravel()
    faces = np.stack([i, i + 1, i + n, i + 1, i + n + 1, i + n], axis=1).reshape(-1)
    meshes = [{"primitives": [{"attributes": {"POSITION": b.accessor(pos, "VEC3", bounds=True)},
                               "indices": b.accessor(faces, "SCALAR", component=5125)}]}]
    return _write(path, meshes, [{"mesh": 0}], [0], b)


def _predicted(path, joints, sparse, seed=0):
    """A rig "predicted" for ``path`` in a normalized frame, and its scene."""
    rng = np.random.default_rng(seed)
    original = GlbParser.load(str(path))
    vertices = original.vertices * 0.5 + 1.0
    lo, hi = vertices.min(0), vertices.max(0)
    heads = lo + (hi - lo) * rng.random((joints, 3))
    parents = np.array([-1] + [int(rng.integers(0, j)) for j in range(1, joints)])
    asset = Asset.from_data(vertices=vertices, joints=heads, parents=parents,
                            joint_names=[f"joint_{j}" for j in range(joints)], skin=None)
    N = len(vertices)
    if sparse:
        ids = np.stack([rng.choice(joints, 4, replace=False) for _ in range(N)])
        skin = SparseSkin(ids, rng.random((N, 4)).astype(np.float32))
    else:
        skin = rng.random((N, joints)).astype(np.float32)
    return asset, skin


CASES = {
    "instances+points": (lambda p: _instances(p), 3, False),
    "skinned input": (lambda p: _skinned(p.parent, 3), 3, False),
    "300 joints (uint16 ids)": (lambda p: _instances(p), 300, True),
    "70k vertices (uint32 indices)": (lambda p: _grid(p, 265), 24, True),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("compress", [True, False], ids=["zlib", "raw"])
def test_rebuilt_glb_is_byte_identical_to_the_server_glb(tmp_path, case, compress):
    make, joints, sparse = CASES[case]
    path = make(tmp_path / "input.glb")
    upload = path.read_bytes()
    asset, skin = _predicted(path, joints, sparse)
    server_scene = load_scene(upload, name="input.glb")

    glb = transfer_rigging(asset, scene=server_scene, skin=skin, auto_ground=True)
    payload = encode_rig_payload(server_scene,
                                 transfer_rig(asset, server_scene, skin=skin, auto_ground=True),
                                 compress=compress)

    assert payload[:4] == MAGIC
    assert build_glb(upload, payload, name="input.glb") == glb
    # The client may parse its own copy of the file any way load_scene takes it.
    assert build_glb(str(path), payload) == glb
    if case.startswith("70k"):
        # 12 of the GLB's ~48 bytes per vertex, before (random, so barely
        # compressible) weights are compressed.
        assert len(payload) < len(glb) / 3


def test_layout_records_the_vertex_order(tmp_path):
    scene = load_scene(_instances(tmp_path / "input.glb"))
    assert scene_layout(scene) == [
        {"node": 1, "mesh": 0, "name": "a", "skinned_input": False,
         "primitives": [{"index": 0, "vertex_count": 3}, {"index": 2, "vertex_count": 4}]},
        {"node": 2, "mesh": 0, "name": "b", "skinned_input": False,
         "primitives": [{"index": 0, "vertex_count": 3}, {"index": 2, "vertex_count": 4}]},
    ]
    skinned = load_scene(_skinned(tmp_path, 2))
    assert [entry["skinned_input"] for entry in scene_layout(skinned)] == [True]


def _payload(tmp_path, **kwargs):
    path = _instances(tmp_path / "input.glb")
    scene = load_scene(path)
    asset, skin = _predicted(path, 3, False)
    return path, encode_rig_payload(scene, transfer_rig(asset, scene, skin=skin, auto_ground=True), **kwargs)


def test_decoded_payload_carries_the_rig(tmp_path):
    path, payload = _payload(tmp_path)
    decoded = decode_rig_payload(payload)
    assert decoded.vertex_count == 14
    assert decoded.rig.joint_names == ("joint_0", "joint_1", "joint_2")
    assert decoded.rig.ids.shape == (14, 4) and decoded.rig.ids.dtype == np.uint8
    assert decoded.rig.weights.dtype == np.uint16
    np.testing.assert_array_equal(decoded.rig.weights.astype(int).sum(axis=1), 65535)


def test_payload_for_another_input_is_refused(tmp_path):
    _, payload = _payload(tmp_path)
    other = _grid(tmp_path / "other.glb", 4)
    with pytest.raises(ValueError, match="does not match this input"):
        build_glb(str(other), payload)


def _split(payload):
    json_len = struct.unpack_from("<I", payload, 12)[0]
    header = json.loads(payload[20:20 + json_len])
    stored = payload[20 + json_len + 8:]
    return header, stored


def _join(header, stored, version=1):
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 4)
    stored += bytes(-len(stored) % 4)
    length = 12 + 8 + len(text) + 8 + len(stored)
    return (MAGIC + struct.pack("<II", version, length) + struct.pack("<II", len(text), 0x4E4F534A)
            + text + struct.pack("<II", len(stored), 0x004E4942) + stored)


def _tampered(payload, edit):
    header, stored = _split(payload)
    edit(header)
    return _join(header, stored)


def _bomb(header):
    header["bin"]["byte_length"] = 16  # the stream inflates far beyond this


@pytest.mark.parametrize("mutate, match", [
    (lambda p: p[:-8], "length"),
    (lambda p: b"glTF" + p[4:], "bad magic"),
    (lambda p: _join(*_split(p), version=2), "unsupported version"),
    (lambda p: _tampered(p, lambda h: h["parents"].__setitem__(1, 7)), "parents"),
    (lambda p: _tampered(p, lambda h: h.__setitem__("vertex_count", 15)), "byte_length"),
    (lambda p: _tampered(p, lambda h: h["joint_names"].pop()), "parents"),
    (lambda p: _tampered(p, lambda h: h["bin"].__setitem__("compression", "gzip")), "compression"),
    (lambda p: _tampered(p, _bomb), "zlib"),
    (lambda p: _tampered(p, lambda h: h["joint_weights"].__setitem__("normalized", False)), "normalized"),
    (lambda p: _tampered(p, lambda h: h.__setitem__("ground_offset", [0, 1])), "3-vectors"),
])
def test_malformed_payloads_are_value_errors(tmp_path, mutate, match):
    _, payload = _payload(tmp_path)
    with pytest.raises(ValueError, match=match):
        decode_rig_payload(mutate(payload))


def test_out_of_range_joint_ids_are_refused(tmp_path):
    _, payload = _payload(tmp_path, compress=False)
    header, stored = _split(payload)
    stored = bytearray(stored)
    stored[0] = 200  # first joint index of the first vertex; there are 3 joints
    with pytest.raises(ValueError, match="joint index out of range"):
        decode_rig_payload(_join(header, bytes(stored)))


def test_corrupt_zlib_data_is_a_value_error(tmp_path):
    _, payload = _payload(tmp_path)
    header, stored = _split(payload)
    stored = bytearray(stored)
    stored[5] ^= 0xFF
    with pytest.raises(ValueError, match="invalid rig payload"):
        decode_rig_payload(_join(header, bytes(stored)))


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
def test_pipeline_exports_rig_payloads_that_rebuild_its_glbs(tmp_path):
    from src.server import stages
    from src.server.config import GenParams
    from src.server.worker import Cancellation

    path = _instances(tmp_path / "input.glb")
    source = stages.Upload(path.read_bytes(), "input.glb")
    results = {}
    for output in ("glb", "rig"):
        asset, skin = _predicted(path, 3, False)
        asset.skin = skin
        results[output] = stages.export_samples(
            [SimpleNamespace(asset=asset)], GenParams(seed=1, output=output), "t",
            Cancellation(), None, source=source)
    assert len(results["rig"]) == 1
    assert build_glb(source.data, results["rig"][0], name="input.glb") == results["glb"][0]


def test_glb_output_is_refused_unless_enabled(monkeypatch):
    monkeypatch.setattr(serve.config, "OUTPUT_FORMATS", ("rig",))
    ws = _FakeWs('{"filename": "a.glb", "output": "glb"}', b"never read")
    message = _run_ws(ws)
    assert 'output "glb"' in message and '"rig"' in message
    assert ws.incoming == [b"never read"]  # refused before the upload
    assert "output" in _run_ws(_FakeWs('{"filename": "a.glb", "output": "zip"}', b"x"))
    assert serve.state.admitted == 0


class _PoolWs(_FakeWs):
    async def send_bytes(self, data):
        self.sent.append(data)


class _FakePool:
    """Either pool: the stages hand on placeholders."""

    def __init__(self):
        self.params = None

    def would_queue(self):
        return False

    async def submit(self, params, upload, **kwargs):
        self.params = params
        stage = params.get("stage", "infer")
        if stage == "prepare":
            return {"infer": b"points", "export": b"scene"}
        if stage == "infer":
            assert upload == b"points"
            return b"predictions"
        assert upload == (b"scene", b"predictions")
        return [b"payload:" + params["output"].encode()]


@pytest.mark.parametrize("formats, header, output", [
    (("rig",), {"output": "rig"}, "rig"),
    (("rig",), {}, "rig"),  # a header without a format asks for a rig payload
    (("rig", "glb"), {"output": "glb"}, "glb"),
])
def test_supported_output_reaches_the_worker_and_the_done_message(monkeypatch, formats, header, output):
    monkeypatch.setattr(serve.config, "OUTPUT_FORMATS", formats)
    serve.state.pool = _FakePool()
    pool = serve.state.cpu_pool = _FakePool()
    ws = _PoolWs(json.dumps({"filename": "a.glb", **header}), b"upload", hang=True)
    asyncio.run(serve.ws_generate(ws))
    assert pool.params["output"] == output
    done = next(m for m in ws.sent if isinstance(m, dict) and m.get("stage") == "done")
    assert done["format"] == output and done["payload_size"] == len(ws.sent[-1])
    assert ws.sent[-1] == b"payload:" + output.encode()


def test_output_formats_follow_the_environment(monkeypatch):
    import importlib
    from src.server import config
    try:
        monkeypatch.delenv("SKINTOKENS_ENABLE_GLB_OUTPUT", raising=False)
        assert importlib.reload(config).OUTPUT_FORMATS == ("rig",)
        monkeypatch.setenv("SKINTOKENS_ENABLE_GLB_OUTPUT", "1")
        assert importlib.reload(config).OUTPUT_FORMATS == ("rig", "glb")
    finally:
        monkeypatch.delenv("SKINTOKENS_ENABLE_GLB_OUTPUT", raising=False)
        importlib.reload(config)


# --------------------------------------------------------------------------- #
# the client's result: the input file with the rig added
# --------------------------------------------------------------------------- #
def _rotation(name):
    """A fixed rotation per joint name: the same pose in both files."""
    from scipy.spatial.transform import Rotation
    seed = int(name.rsplit("_", 1)[-1]) if name.rsplit("_", 1)[-1].isdigit() else 0
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(np.random.default_rng(seed).normal(size=3) * 0.4).as_matrix()
    return m


def _skinned_vertices(glb, primitives, posed):
    """Vertices skinned by the glTF rules (a skinned mesh's node transform is
    ignored); ``primitives(data)`` lists (node, primitive indices)."""
    from src.rig_package.glb_io import Document
    from src.rig_package.glb_io.document import node_local_matrix
    doc = Document(glb, name="out.glb")
    data, nodes = doc.data, doc.data["nodes"]
    joints = {j for skin in data["skins"] for j in skin["joints"]}
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    world = {}

    def world_of(i):
        if i not in world:
            local = node_local_matrix(nodes[i])
            if posed and i in joints:
                local = local @ _rotation(nodes[i]["name"])
            world[i] = (world_of(parent[i]) if i in parent else np.eye(4)) @ local
        return world[i]

    out = []
    for node_index, prim_indices in primitives(data):
        node = nodes[node_index]
        skin = data["skins"][node["skin"]]
        ibm = doc.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        M = np.stack([world_of(j) for j in skin["joints"]]) @ ibm
        for p in prim_indices:
            attrs = data["meshes"][node["mesh"]]["primitives"][p]["attributes"]
            positions = doc.accessor(attrs["POSITION"])
            v = np.c_[positions, np.ones(len(positions))]
            ids = doc.accessor(attrs["JOINTS_0"]).astype(int)
            w = doc.accessor(attrs["WEIGHTS_0"])
            out.append(np.einsum("nk,nkij,nj->ni", w, M[ids], v)[:, :3])
    return np.concatenate(out)


def _server_primitives(data):
    return [(i, range(len(data["meshes"][n["mesh"]]["primitives"])))
            for i, n in enumerate(data["nodes"]) if "mesh" in n]


def _gltf_embedded(glb_path):
    """The GLB as a .gltf with its buffer embedded as a data URI."""
    import base64
    from src.rig_package.glb_io import Document
    doc = Document(glb_path)
    data = json.loads(json.dumps(doc.data))
    data["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
        doc.binary[:data["buffers"][0]["byteLength"]]).decode()
    out = glb_path.with_suffix(".gltf")
    out.write_text(json.dumps(data))
    return out


APPLY_CASES = {
    "instances+points": (lambda p: _instances(p), 5, False),
    "skinned input": (lambda p: _skinned(p.parent, 3), 3, False),
    "300 joints": (lambda p: _instances(p), 300, True),
    "embedded .gltf": (lambda p: _gltf_embedded(_instances(p)), 5, False),
}


@pytest.mark.parametrize("case", list(APPLY_CASES))
@pytest.mark.parametrize("posed", [False, True], ids=["rest", "posed"])
def test_rigged_input_skins_like_the_server_glb(tmp_path, case, posed):
    from src.rig_package.glb_io import apply_rig
    make, joints, sparse = APPLY_CASES[case]
    path = make(tmp_path / "input.glb")
    upload = path.read_bytes()
    name = "input" + path.suffix
    asset, skin = _predicted(path, joints, sparse)
    scene = load_scene(upload, name=name)
    rig = transfer_rig(asset, scene, skin=skin, auto_ground=True)
    payload = encode_rig_payload(scene, rig)
    server_glb = build_glb(upload, payload, name=name)
    rigged = apply_rig(upload, payload, name=name)

    layout = scene_layout(scene)
    ours = _skinned_vertices(rigged, lambda data: [
        (e["node"], [p["index"] for p in e["primitives"]]) for e in layout], posed)
    # The server's GLB stands on the ground; the rigged input keeps its place.
    theirs = _skinned_vertices(server_glb, _server_primitives, posed) - rig.offset
    np.testing.assert_allclose(ours, theirs, atol=1e-5 * max(1.0, float(np.ptp(theirs))))


def test_rigged_input_keeps_everything_the_input_has():
    """A textured file: materials, textures, images, UVs, nodes and every
    accessor's data come through unchanged; the rig is only added."""
    from src.rig_package.glb_io import Document, apply_rig
    path = "examples/giraffe.glb"
    upload = Path(path).read_bytes()
    asset, skin = _predicted(path, 12, True)
    scene = load_scene(upload, name="input.glb")
    payload = encode_rig_payload(scene, transfer_rig(asset, scene, skin=skin, auto_ground=True))
    before = Document(upload, name="a.glb")
    after = Document(apply_rig(upload, payload, name="input.glb"), name="b.glb")
    b, a = before.data, after.data
    assert b.get("images") and b.get("materials") and b.get("textures")
    for key in set(b) - {"nodes", "meshes", "accessors", "bufferViews", "buffers", "scenes", "skins"}:
        assert a[key] == b[key], key
    for i in range(len(b["accessors"])):
        np.testing.assert_array_equal(after.accessor(i), before.accessor(i))
    for image in b["images"]:
        if "bufferView" in image:
            assert after.view_bytes(image["bufferView"]) == before.view_bytes(image["bufferView"])
    for i, node in enumerate(b["nodes"]):
        assert {k: v for k, v in a["nodes"][i].items() if k != "skin"} == node
    assert len(a["nodes"]) == len(b["nodes"]) + 1 + 12  # armature + joints
    for mesh_before, mesh_after in zip(b["meshes"], a["meshes"]):
        for pb, pa in zip(mesh_before["primitives"], mesh_after["primitives"]):
            assert {k: v for k, v in pa.items() if k != "attributes"} == \
                   {k: v for k, v in pb.items() if k != "attributes"}  # material etc.
            assert set(pa["attributes"]) == set(pb["attributes"]) | {"JOINTS_0", "WEIGHTS_0"}
            assert all(pa["attributes"][k] == v for k, v in pb["attributes"].items())


def test_rigged_input_copies_shared_meshes_and_rigs_every_primitive(tmp_path):
    from src.rig_package.glb_io import Document, apply_rig
    path = _instances(tmp_path / "input.glb")
    asset, skin = _predicted(path, 5, False)
    scene = load_scene(path)
    payload = encode_rig_payload(scene, transfer_rig(asset, scene, skin=skin))
    data = Document(apply_rig(str(path), payload), name="out.glb").data
    assert [data["nodes"][i]["mesh"] for i in (1, 2)] == [0, 1]  # node b got a copy
    assert data["meshes"][1]["name"] == "body"
    for mesh in data["meshes"]:
        # The points primitive is rigged too (nearest rigged vertex).
        assert all({"JOINTS_0", "WEIGHTS_0"} <= set(p["attributes"]) for p in mesh["primitives"])
        assert mesh["primitives"][0]["attributes"]["POSITION"] == 0  # geometry shared
    assert data["nodes"][1]["skin"] != data["nodes"][2]["skin"]  # placed differently


def test_rigged_gltf_moves_its_embedded_buffers_into_the_glb(tmp_path):
    from src.rig_package.glb_io import Document, apply_rig
    path = _gltf_embedded(_instances(tmp_path / "input.glb"))
    asset, skin = _predicted(path, 5, False)
    scene = load_scene(path)
    payload = encode_rig_payload(scene, transfer_rig(asset, scene, skin=skin))
    before, after = Document(path), Document(apply_rig(str(path), payload), name="out.glb")
    assert len(after.data["buffers"]) == 1 and "uri" not in after.data["buffers"][0]
    for i in range(len(before.data["accessors"])):
        np.testing.assert_array_equal(after.accessor(i), before.accessor(i))


def test_rigged_skinned_input_replaces_its_old_skin(tmp_path):
    from src.rig_package.glb_io import Document, apply_rig
    path = _skinned(tmp_path, 3)
    asset, skin = _predicted(path, 4, False)
    scene = load_scene(path)
    payload = encode_rig_payload(scene, transfer_rig(asset, scene, skin=skin))
    doc = Document(apply_rig(str(path), payload), name="out.glb")
    # The input's own skin is unreferenced now, so it is dropped.
    assert len(doc.data["skins"]) == 1 and doc.data["nodes"][0]["skin"] == 0
    ids = doc.accessor(doc.data["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"])
    np.testing.assert_array_equal(ids, decode_rig_payload(payload).rig.ids)


def test_rigged_input_skips_primitives_without_vertices(tmp_path):
    from src.rig_package.glb_io import Document, apply_rig, prepare_skinned_rig
    b = Builder()
    tri = b.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), "VEC3", bounds=True)
    empty = b.accessor(np.zeros((0, 3)), "VEC3", bounds=True)
    b.data["meshes"] = [{"primitives": [
        {"attributes": {"POSITION": tri}},
        {"attributes": {"POSITION": empty}, "mode": 0}]}]
    b.data["nodes"] = [{"mesh": 0}]
    b.data["scenes"] = [{"nodes": [0]}]
    path = tmp_path / "empty_point.glb"
    b.write(path)

    scene = load_scene(path)
    skin = SparseSkin(np.zeros((3, 4), int), np.full((3, 4), 0.25, np.float32))
    payload = encode_rig_payload(scene, prepare_skinned_rig(
        scene, np.zeros((2, 3)), [-1, 0], ["a", "b"], skin))
    before = Document(path)
    doc = Document(apply_rig(str(path), payload), name="out.glb")
    rigged, empty_point = doc.data["meshes"][0]["primitives"]
    assert {"JOINTS_0", "WEIGHTS_0"} <= set(rigged["attributes"])
    # Nothing to skin: no empty accessors are added, which glTF forbids.
    assert set(empty_point["attributes"]) == {"POSITION"}
    added = doc.data["accessors"][len(before.data["accessors"]):]
    assert added and all(a["count"] for a in added)


def test_more_influences_than_the_format_carries_are_refused(tmp_path):
    from src.rig_package.glb_io import MAX_INFLUENCES, prepare_skinned_rig
    path = _instances(tmp_path / "input.glb")
    scene = load_scene(path)
    n = scene.vertex_count
    wide = SparseSkin(np.zeros((n, MAX_INFLUENCES + 4), int),
                      np.full((n, MAX_INFLUENCES + 4), 0.25, np.float32))
    rig = prepare_skinned_rig(scene, np.zeros((2, 3)), [-1, 0], ["a", "b"], wide,
                              group_per_vertex=None)
    with pytest.raises(ValueError, match="influences per vertex"):
        encode_rig_payload(scene, rig)
