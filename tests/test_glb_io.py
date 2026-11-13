"""Native GLB I/O: load, rig transfer and export without bpy."""

import builtins
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.rig_package.glb_io import Builder, Document, Y_UP_TO_Z_UP, load_scene
from src.rig_package.info.asset import Asset
from src.rig_package.parser.glb import GlbParser, transfer_rigging

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def _block_bpy(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("bpy", "mathutils"):
            raise AssertionError(f"native GLB I/O imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _gltf_json(path):
    raw = Path(path).read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    return json.loads(raw[20:20 + length])


def _read_rig(path):
    """Exported rig as (names, parents, Z-up joint heads, dense (N, J) weights)."""
    doc = Document(path)
    data = doc.data
    skin = data["skins"][0]
    joints = skin["joints"]
    index = {n: j for j, n in enumerate(joints)}
    parents = np.full(len(joints), -1)
    for n in joints:
        for c in data["nodes"][n].get("children", []):
            parents[index[c]] = index[n]
    ibm = doc.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    heads = np.linalg.inv(ibm)[:, :3, 3] @ Y_UP_TO_Z_UP.T
    weights = []
    for node in data["nodes"]:
        if "mesh" not in node:
            continue
        assert node["skin"] == 0
        for prim in data["meshes"][node["mesh"]]["primitives"]:
            ids = doc.accessor(prim["attributes"]["JOINTS_0"]).astype(int)
            w = doc.accessor(prim["attributes"]["WEIGHTS_0"])
            dense = np.zeros((len(ids), len(joints)))
            np.add.at(dense, (np.repeat(np.arange(len(ids)), 4), ids.ravel()), w.ravel())
            weights.append(dense)
    names = [data["nodes"][n]["name"] for n in joints]
    return names, parents, heads, np.concatenate(weights)


def _write_transformed_mesh(path):
    """Two instances of one triangle under a mirrored, scaled, translated parent."""
    b = Builder()
    pos = b.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), "VEC3", bounds=True)
    nrm = b.accessor(np.array([[0, 0, 1]] * 3, float), "VEC3")
    idx = b.accessor(np.array([0, 1, 2]), "SCALAR", component=5125)
    b.data["meshes"] = [{"primitives": [{"attributes": {"POSITION": pos, "NORMAL": nrm}, "indices": idx}]}]
    b.data["nodes"] = [
        {"name": "parent", "translation": [0, 2, 0], "scale": [-2, 2, 2], "children": [1, 2]},
        {"name": "a", "mesh": 0},
        {"name": "b", "mesh": 0, "translation": [0, 0, 5]},
    ]
    b.data["scenes"] = [{"nodes": [0]}]
    b.write(path)


@pytest.mark.parametrize("name", ["giraffe.glb", "horse_mesh_only.glb"])
def test_transfer_roundtrip_preserves_mesh_and_rig(tmp_path, name):
    src_path = EXAMPLES / name
    original = GlbParser.load(str(src_path))
    assert original.N > 0 and original.F > 0

    # A rig predicted in a normalized space (similarity of the original mesh).
    normalized = original.vertices * 0.5 + 1.0
    lo, hi = normalized.min(0), normalized.max(0)
    joints = np.stack([lo + (hi - lo) * np.array([0.5, 0.5, t]) for t in (0.2, 0.5, 0.8)])
    skin = np.random.default_rng(0).random((original.N, 3)).astype(np.float32)
    predicted = Asset.from_data(vertices=normalized, joints=joints, parents=np.array([-1, 0, 1]),
                                joint_names=["root", "mid", "tip"], skin=skin)

    out = tmp_path / "rigged.glb"
    transfer_rigging(predicted, str(src_path), str(out), group_per_vertex=4, auto_ground=True)
    rigged = GlbParser.load(str(out))
    names, parents, heads, weights = _read_rig(out)

    ground = original.vertices[:, 2].min()
    expected_vertices = original.vertices - [0, 0, ground]
    np.testing.assert_allclose(rigged.vertices, expected_vertices, atol=1e-5)
    np.testing.assert_array_equal(rigged.faces, original.faces)
    assert names == ["root", "mid", "tip"]
    np.testing.assert_array_equal(parents, [-1, 0, 1])
    expected_joints = (joints - 1.0) / 0.5 - [0, 0, ground]
    np.testing.assert_allclose(heads, expected_joints, atol=1e-4 * np.ptp(original.vertices))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)

    # Output is the skinned mesh only: no materials/textures/images/UVs.
    out_json = _gltf_json(out)
    for key in ("materials", "textures", "samplers", "images", "extensionsUsed", "extensionsRequired"):
        assert key not in out_json
    assert len(out_json["skins"]) == 1
    for mesh in out_json["meshes"]:
        for prim in mesh["primitives"]:
            assert set(prim) <= {"attributes", "indices", "mode"}
            assert set(prim["attributes"]) <= {"POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0"}


def test_export_asset_writes_rig_and_skinned_input_is_plain_mesh(tmp_path):
    original = GlbParser.load(str(EXAMPLES / "giraffe.glb"))
    joints = np.array([[0, 0, 1.0], [0, 0, 2.0], [0, -1.0, 3.0]])
    skin = np.zeros((original.N, 3), np.float32)
    skin[:, 1] = 1
    asset = Asset.from_data(vertices=original.vertices, faces=original.faces, joints=joints,
                            parents=np.array([-1, 0, 1]), joint_names=["Hips", "Spine", "Head"], skin=skin)
    out = tmp_path / "skinned.glb"
    GlbParser.export_asset(asset, str(out))

    names, parents, heads, weights = _read_rig(out)
    assert names == ["Hips", "Spine", "Head"]
    np.testing.assert_array_equal(parents, [-1, 0, 1])
    np.testing.assert_allclose(heads, joints, atol=1e-6)
    np.testing.assert_allclose(weights, skin, atol=1e-6)

    # A skinned file used as input is just a mesh: its rig is ignored.
    loaded = GlbParser.load(str(out))
    assert loaded.joint_names is None and loaded.parents is None
    assert loaded.matrix_local is None and loaded.skin is None
    np.testing.assert_allclose(loaded.vertices, original.vertices, atol=1e-6)


def test_node_transforms_are_baked_with_blender_axes(tmp_path):
    path = tmp_path / "transformed.glb"
    _write_transformed_mesh(path)
    scene = load_scene(path)
    assert [inst.name for inst in scene.instances] == ["a", "b"]

    asset = GlbParser.load(str(path))
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    world_a = tri * [-2, 2, 2] + [0, 2, 0]
    world_b = (tri + [0, 0, 5]) * [-2, 2, 2] + [0, 2, 0]
    expected = np.concatenate([world_a, world_b]) @ Y_UP_TO_Z_UP.T
    np.testing.assert_allclose(asset.vertices, expected, atol=1e-6)
    # The mirrored parent flips winding so normals still face +glTF Z (Blender -Y).
    assert np.all(asset.face_normals[:, 1] < -0.99)

    predicted = Asset.from_data(vertices=asset.vertices, joints=np.zeros((1, 3)), parents=np.array([-1]),
                                joint_names=["root"], skin=np.ones((asset.N, 1), np.float32))
    out = tmp_path / "rigged.glb"
    transfer_rigging(predicted, str(path), str(out))
    rigged = GlbParser.load(str(out))
    np.testing.assert_allclose(rigged.vertices, asset.vertices, atol=1e-6)
    np.testing.assert_allclose(rigged.face_normals, asset.face_normals, atol=1e-6)
    normals = _gltf_json(out)  # baked NORMAL accessors exist and are not stale
    assert all("NORMAL" in p["attributes"] for m in normals["meshes"] for p in m["primitives"])


def test_fbx_is_rejected(tmp_path):
    path = tmp_path / "model.fbx"
    path.write_bytes(b"Kaydara FBX Binary")
    with pytest.raises(ValueError, match="FBX"):
        GlbParser.load(str(path))


@pytest.mark.parametrize('mode', [5, 6])
def test_triangle_strip_and_fan_export_as_triangles(tmp_path, mode):
    b = Builder()
    pos = b.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), 'VEC3')
    b.data['meshes'] = [{'primitives': [{'attributes': {'POSITION': pos}, 'mode': mode}]}]
    b.data['nodes'] = [{'mesh': 0}]
    b.data['scenes'] = [{'nodes': [0]}]
    path = tmp_path / 'input.glb'
    b.write(path)
    asset = GlbParser.load(str(path))
    assert asset.N == 4 and asset.F == 2
    predicted = Asset.from_data(vertices=asset.vertices, joints=np.zeros((1, 3)),
                                parents=np.array([-1]), joint_names=['root'],
                                skin=np.ones((asset.N, 1), np.float32))
    out = tmp_path / 'out.glb'
    transfer_rigging(predicted, str(path), str(out))
    np.testing.assert_array_equal(GlbParser.load(str(out)).faces, asset.faces)
    assert _gltf_json(out)['meshes'][0]['primitives'][0].get('mode', 4) == 4


def test_obj_input_loads_and_rigs(tmp_path):
    obj = tmp_path / "quad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n")
    asset = GlbParser.load(str(obj))
    assert asset.N == 4 and asset.F == 2
    # OBJ is Y-up like glTF: (x, y, z) -> Blender (x, -z, y)
    np.testing.assert_allclose(np.sort(asset.vertices[:, 2]), [0, 0, 1, 1])
    predicted = Asset.from_data(vertices=asset.vertices, joints=np.zeros((1, 3)), parents=np.array([-1]),
                                joint_names=["root"], skin=np.ones((asset.N, 1), np.float32))
    out = tmp_path / "quad.glb"
    transfer_rigging(predicted, str(obj), str(out))
    assert _read_rig(out)[0] == ["root"]
