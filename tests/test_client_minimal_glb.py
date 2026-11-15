"""Client upload minimization must preserve the rig payload's vertex layout."""
import numpy as np
import pytest

from skintokens_client import _minimal_glb_for_rig
from src.rig_package.glb_io import Builder, Document, load_scene, scene_layout


def _scene_builder():
    b = Builder()
    pos = b.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                     "VEC3", bounds=True)
    uv = b.accessor(np.zeros((4, 2)), "VEC2")
    idx = b.accessor(np.array([0, 1, 2]), "SCALAR", component=5123)
    b.data["meshes"] = [{"name": "mesh", "primitives": [
        {"mode": 1, "attributes": {"POSITION": pos}},
        {"attributes": {"POSITION": pos, "TEXCOORD_0": uv}, "indices": idx},
        {"mode": 0, "attributes": {"POSITION": pos}},
        {"mode": 5, "attributes": {"POSITION": pos}},
        {"mode": 6, "attributes": {"POSITION": pos}},
    ]}]
    b.data["nodes"] = [
        {"translation": [0, 2, 0], "scale": [-2, 2, 2], "children": [2, 1]},
        {"name": "shared", "mesh": 0, "translation": [0, 0, 5]},
        {"name": "shared", "mesh": 0, "skin": 0, "translation": [10, 0, 0]},
        {"name": "joint"},
    ]
    b.data["skins"] = [{"joints": [3]}]
    b.data["scenes"] = [{"nodes": [3]}, {"nodes": [0, 3]}]
    b.data["scene"] = 1
    return b


@pytest.mark.parametrize("scenes", ["default", "absent", "empty"])
def test_minimal_upload_preserves_scene_and_rig_layout(scenes):
    b = _scene_builder()
    if scenes == "absent":
        b.data.pop("scenes")
        b.data.pop("scene")
    elif scenes == "empty":
        b.data["scenes"] = []
        b.data.pop("scene")
    original = load_scene(b.to_bytes())
    minimized = load_scene(_minimal_glb_for_rig(b.to_bytes()))

    assert scene_layout(minimized) == scene_layout(original)
    assert minimized.order == original.order
    assert [i.node for i in minimized.instances] == [2, 1]
    for before, after in zip(original.instances, minimized.instances):
        assert [p.index for p in after.primitives] == [1, 3, 4]
        np.testing.assert_array_equal(after.world, before.world)
        for p, q in zip(before.primitives, after.primitives):
            np.testing.assert_array_equal(q.positions, p.positions)
            np.testing.assert_array_equal(q.faces, p.faces)
    for mesh in minimized.document.data["meshes"]:
        for primitive in mesh["primitives"]:
            assert set(primitive.get("attributes", {})) <= {"POSITION"}


def test_draco_rejected_before_decoding_accessors(monkeypatch):
    b = _scene_builder()
    b.data["meshes"][0]["primitives"][1]["extensions"] = {
        "KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {"POSITION": 0}}
    }
    raw = b.to_bytes()
    message = "KHR_draco_mesh_compression meshes are not supported; export the GLB without Draco compression"
    with pytest.raises(ValueError, match=message):
        load_scene(raw)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Draco must be rejected before reading any accessor")

    monkeypatch.setattr(Document, "accessor", unexpected_decode)
    with pytest.raises(ValueError, match=message):
        _minimal_glb_for_rig(raw)


@pytest.mark.parametrize("location", ["extensionsUsed", "extensionsRequired", "node"])
def test_gpu_instancing_rejected_before_decoding_accessors(location, monkeypatch):
    b = _scene_builder()
    extension = "EXT_mesh_gpu_instancing"
    if location == "node":
        b.data["nodes"][1]["extensions"] = {extension: {"attributes": {}}}
    else:
        b.data[location] = [extension]
    raw = b.to_bytes()
    message = extension + " is not supported; expand mesh instances before importing"
    with pytest.raises(ValueError, match=message):
        load_scene(raw)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("GPU instancing must be rejected before reading any accessor")

    monkeypatch.setattr(Document, "accessor", unexpected_decode)
    with pytest.raises(ValueError, match=message):
        _minimal_glb_for_rig(raw)
