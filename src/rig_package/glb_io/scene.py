"""Mesh view of a glTF scene and native skinned-GLB export.

Inputs are treated as plain meshes: any skin, joints or animation in them are
ignored; rigs are only output. Geometry is reported in glTF world space (Y-up).
A mesh that happens to be skinned is read in its bind pose (its vertices as
stored; glTF ignores a skinned mesh's node transform).

Callers that need Blender's Z-up convention (the space the model was trained
in) convert with :data:`Y_UP_TO_Z_UP`, exactly like Blender's glTF importer.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .builder import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, Builder
from .document import Document, malformed_as_value_error, node_local_matrix
from .limits import InputLimits, check_limit
from ..skin_interp import clean_skin, select_top_k

# glTF (x, y, z) Y-up  ->  Blender (x, -z, y) Z-up.
Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
Z_UP_TO_Y_UP = Y_UP_TO_Z_UP.T

SUPPORTED_SCENE_EXT = ('.glb', '.gltf')


def _is_triangle_primitive(prim: dict) -> bool:
    mode = prim.get('mode', 4)
    if mode not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError(f'Unsupported primitive mode: {mode}')
    return mode in (4, 5, 6)


@dataclass
class Primitive:
    json: dict
    # (n, 3) float64 vertex positions in world/bind space (glTF Y-up)
    positions: np.ndarray
    # (f, 3) triangle indices local to this primitive, winding already corrected
    # for mirrored node transforms
    faces: np.ndarray


@dataclass
class MeshInstance:
    node: int
    name: str
    mesh: int
    skin: Optional[int]
    # transform baked into ``Primitive.positions`` (identity for skinned meshes)
    world: np.ndarray
    primitives: List[Primitive] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return sum(len(p.positions) for p in self.primitives)


class Scene:
    """Flattened mesh instances of the default glTF scene, in traversal order."""

    def __init__(self, document: Document):
        self.document = document
        with malformed_as_value_error(document.path.name):
            self._parse()

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    def _parse(self):
        data = self.document.data
        nodes = data.get('nodes', [])
        extension = 'EXT_mesh_gpu_instancing'
        if (extension in data.get('extensionsUsed', [])
                or extension in data.get('extensionsRequired', [])
                or any(extension in node.get('extensions', {}) for node in nodes)):
            raise ValueError(f'{extension} is not supported; expand mesh instances before importing')
        self.nodes = nodes
        self.parent = np.full(len(nodes), -1, dtype=int)
        for i, node in enumerate(nodes):
            for child in node.get('children', []):
                if child < 0 or child >= len(nodes) or self.parent[child] != -1 or child == i:
                    raise ValueError('Invalid glTF node hierarchy')
                self.parent[child] = i
        scenes = data.get('scenes')
        if scenes:
            roots = list(scenes[data.get('scene', 0)].get('nodes', []))
        else:
            roots = [int(i) for i in np.flatnonzero(self.parent < 0)]
        self.order: List[int] = []
        self.world: Dict[int, np.ndarray] = {}
        visited = set()
        stack = [(r, np.eye(4)) for r in reversed(roots)]
        while stack:
            i, parent_world = stack.pop()
            if i in visited:
                raise ValueError('Cyclic glTF hierarchy')
            visited.add(i)
            self.order.append(i)
            self.world[i] = parent_world @ node_local_matrix(nodes[i])
            for child in reversed(nodes[i].get('children', [])):
                stack.append((child, self.world[i]))

        if self.document.limits is not None:
            self._check_mesh_limits(self.document.limits)
        self.instances: List[MeshInstance] = []
        used_names = set()
        for i in self.order:
            node = nodes[i]
            if 'mesh' not in node:
                continue
            skin = node.get('skin')
            world = np.eye(4) if skin is not None else self.world[i]
            name = node.get('name') or data['meshes'][node['mesh']].get('name') or f'mesh_{len(self.instances)}'
            base, k = name, 1
            while name in used_names:
                name = f'{base}.{k:03d}'
                k += 1
            used_names.add(name)
            inst = MeshInstance(node=i, name=name, mesh=node['mesh'], skin=skin, world=world)
            for prim in data['meshes'][node['mesh']]['primitives']:
                # Skip points and lines before reading any accessor metadata/data.
                if _is_triangle_primitive(prim):
                    parsed = self._primitive(prim, world)
                    if len(parsed.faces):
                        inst.primitives.append(parsed)
            if inst.primitives:
                self.instances.append(inst)
        if not self.instances:
            raise ValueError('No triangle geometry found in input (points and lines are ignored)')

    def _check_mesh_limits(self, limits: InputLimits):
        """Bound decoded geometry from accessor metadata, before decoding any of it."""
        data = self.document.data
        accessors = data.get('accessors', [])
        mesh_nodes = [i for i in self.order if 'mesh' in self.nodes[i]]
        check_limit(len(mesh_nodes), limits.max_mesh_instances, 'mesh instance count')
        vertices = faces = primitives = 0
        for i in mesh_nodes:
            mesh_primitives = data['meshes'][self.nodes[i]['mesh']]['primitives']
            primitives += len(mesh_primitives)
            check_limit(primitives, limits.max_primitives, 'primitive count')
            for prim in mesh_primitives:
                if not _is_triangle_primitive(prim):
                    continue
                count = int(accessors[prim['attributes']['POSITION']]['count'])
                vertices += count
                elements = int(accessors[prim['indices']]['count']) if 'indices' in prim else count
                faces += elements if prim.get('mode', 4) != 4 else elements // 3
            check_limit(vertices, limits.max_vertices, 'vertex count')
            check_limit(faces, limits.max_faces, 'triangle count')

    def _primitive(self, prim: dict, world: np.ndarray) -> Primitive:
        if 'KHR_draco_mesh_compression' in prim.get('extensions', {}):
            raise ValueError('KHR_draco_mesh_compression meshes are not supported; '
                             'export the GLB without Draco compression')
        attrs = prim['attributes']
        if 'POSITION' not in attrs:
            raise ValueError('glTF primitive without POSITION is not supported')
        local = self.document.accessor(attrs['POSITION']).astype(np.float64)
        positions = local @ world[:3, :3].T + world[:3, 3]
        n = len(positions)
        mode = prim.get('mode', 4)
        if 'indices' in prim:
            idx = self.document.accessor(prim['indices'])[:, 0].astype(np.int64)
            if len(idx) and (idx.min() < 0 or idx.max() >= n):
                raise ValueError('Primitive index out of range')
        else:
            idx = np.arange(n, dtype=np.int64)
        faces = _triangulate(idx, mode)
        if np.linalg.det(world[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]
        return Primitive(json=prim, positions=positions, faces=faces)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    @property
    def vertex_count(self) -> int:
        return sum(inst.vertex_count for inst in self.instances)

    def vertices(self) -> np.ndarray:
        """All vertices (instance, primitive, vertex order), glTF Y-up."""
        arrays = [p.positions for inst in self.instances for p in inst.primitives]
        return np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 3))

    def parts(self):
        """Per instance: ``(name, vertices (n,3), faces (f,3) local to the instance)``."""
        for inst in self.instances:
            verts, faces, offset = [], [], 0
            for p in inst.primitives:
                verts.append(p.positions)
                faces.append(p.faces + offset)
                offset += len(p.positions)
            yield (inst.name,
                   np.concatenate(verts, axis=0) if verts else np.zeros((0, 3)),
                   np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64))



def _triangulate(idx: np.ndarray, mode: int) -> np.ndarray:
    if mode == 4:
        faces = idx[:len(idx) // 3 * 3].reshape(-1, 3)
    elif len(idx) < 3:
        faces = np.zeros((0, 3), dtype=np.int64)
    elif mode == 5:  # strip: p_i = {v_i, v_{i+1+i%2}, v_{i+2-i%2}}
        i = np.arange(len(idx) - 2)
        faces = np.stack([idx[i], idx[i + 1 + i % 2], idx[i + 2 - i % 2]], axis=1)
    else:            # fan:   p_i = {v_{i+1}, v_{i+2}, v_0}
        i = np.arange(len(idx) - 2)
        faces = np.stack([idx[i + 1], idx[i + 2], np.full_like(i, idx[0])], axis=1)
    degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    return faces[~degenerate].astype(np.int64)


# ---------------------------------------------------------------------- #
# loading
# ---------------------------------------------------------------------- #
def load_scene(filepath, limits: Optional[InputLimits] = None, name: Optional[str] = None) -> Scene:
    """Load a .glb/.gltf scene. Pass ``limits`` for untrusted (uploaded) input.

    ``filepath`` may also be the file's bytes, with ``name`` giving its
    extension (an in-memory glTF must embed its resources).
    """
    if isinstance(filepath, (bytes, bytearray, memoryview)):
        name = name or 'memory.glb'
        if Path(name).suffix.lower() not in SUPPORTED_SCENE_EXT:
            raise ValueError(f'unsupported type: {Path(name).suffix}; expected one of {SUPPORTED_SCENE_EXT}')
        return Scene(Document(bytes(filepath), name=name, limits=limits))
    path = Path(filepath)
    if not path.exists():
        raise RuntimeError(f'file does not exist: {filepath}')
    suffix = path.suffix.lower()
    if suffix in ('.glb', '.gltf'):
        return Scene(Document(path, limits=limits))
    if suffix == '.fbx':
        raise ValueError('FBX input is not supported; convert the file to GLB first')
    raise ValueError(f'unsupported type: {suffix}; expected one of {SUPPORTED_SCENE_EXT}')


# ---------------------------------------------------------------------- #
# export
# ---------------------------------------------------------------------- #
@dataclass
class SparseSkin:
    """Each vertex's heaviest influences, as :func:`interpolate_top_k` returns
    them: ``ids`` and ``weights`` are ``(N, k)``, heaviest first, weights
    finite and non-negative (not yet normalized)."""
    ids: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, rows: np.ndarray) -> 'SparseSkin':
        return SparseSkin(self.ids[rows], self.weights[rows])


def _sparsify(skin: np.ndarray, group_per_vertex: int, fallback_joint: int, normalize: bool = True):
    """Top-k influences per vertex, padded to a multiple of 4 slots."""
    skin = np.asarray(skin)
    if not np.issubdtype(skin.dtype, np.floating):
        skin = skin.astype(np.float64)
    J = skin.shape[1]
    k = J if group_per_vertex is None or group_per_vertex <= 0 else min(group_per_vertex, J)
    ids, w = select_top_k(clean_skin(skin), k)
    return _finish_sparse(ids, w, fallback_joint, normalize)


def _finish_sparse(ids: np.ndarray, w: np.ndarray, fallback_joint: int, normalize: bool):
    """Glue for selected influences: fallback for empty rows, normalization,
    unused slots on joint 0, padding to a multiple of 4 slots."""
    k = ids.shape[1]
    ids = np.array(ids, dtype=np.int64)
    w = np.array(w, dtype=np.float64)
    total = w.sum(axis=1)
    empty = total <= 1e-8
    # glTF requires every skinned vertex to have nonzero weight.
    w[empty] = 0.
    ids[empty, 0] = fallback_joint
    w[empty, 0] = 1.
    if normalize:
        w /= w.sum(axis=1, keepdims=True)
    ids[w <= 0] = 0
    pad = -k % 4
    if pad:
        ids = np.pad(ids, ((0, 0), (0, pad)))
        w = np.pad(w, ((0, 0), (0, pad)))
    return ids, w


def _quantize_weights(weights: np.ndarray) -> np.ndarray:
    """Largest-remainder UNORM16 weights, summing exactly to 65535 per vertex.

    Quantize all WEIGHTS_n sets together, preserving zeros and bounding each
    component's absolute error by one quantization step.
    """
    maximum = 65535
    scaled = weights / weights.sum(axis=1, keepdims=True) * maximum
    quantized = np.floor(scaled).astype(np.int64)
    remaining = maximum - quantized.sum(axis=1)
    order = np.argsort(-(scaled - quantized), axis=1, kind='stable')
    rows, ranks = np.nonzero(np.arange(weights.shape[1])[None, :] < remaining[:, None])
    quantized[rows, order[rows, ranks]] += 1
    return quantized.astype(np.uint16)


def _write_attribute(builder: Builder, values: np.ndarray, *, target=ARRAY_BUFFER, bounds=False,
                     normalized=False):
    kind = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4'}[values.shape[1]]
    if values.dtype == np.uint8:
        component = 5121
    elif values.dtype == np.uint16:
        component = 5123
    elif values.dtype == np.uint32:
        component = 5125
    else:
        component = 5126
    if values.shape[1] == 1:
        values = values[:, 0]
    return builder.accessor(values, kind, component=component, bounds=bounds, target=target,
                            normalized=normalized)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.)


def export_skinned_glb(
    scene: Scene,
    filepath,
    joints: np.ndarray,
    parents: Sequence[int],
    joint_names: Sequence[str],
    skin: np.ndarray,
    *,
    group_per_vertex: int = 4,
    normalize: bool = True,
    weight_bits: int = 16,
    offset: Optional[np.ndarray] = None,
    armature_name: str = 'Armature',
):
    """Write ``scene``'s meshes skinned to a new armature as one GLB.

    Args:
        joints: (J, 3) bind-pose joint heads in glTF world space (Y-up).
        skin: (N, J) weights for ``scene.vertices()`` in the same order, or
            their already selected top influences as a :class:`SparseSkin`.
        offset: optional (3,) translation applied to meshes and joints together.
        weight_bits: 16 for normalized integer weights, 32 for float32.
            Defaults to 16. With normalize=False, float32 preserves the sums.

    Output is the skinned mesh only: positions, normals, triangle
    indices and joint weights, baked into bind space (glTF ignores a skinned
    mesh's node transform). Materials, textures, images, UVs, vertex colors,
    tangents, morph targets, extensions, previous skins, animations, cameras
    and lights are not exported.
    """
    with malformed_as_value_error(scene.document.path.name):
        _export_skinned_glb(scene, filepath, joints, parents, joint_names, skin,
                            group_per_vertex=group_per_vertex, normalize=normalize,
                            weight_bits=weight_bits, offset=offset, armature_name=armature_name)


def _export_skinned_glb(scene, filepath, joints, parents, joint_names, skin, *,
                        group_per_vertex, normalize, weight_bits, offset, armature_name):
    if weight_bits not in (16, 32):
        raise ValueError('weight_bits must be 16 or 32')
    document = scene.document
    joints = np.asarray(joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    J = len(joints)
    if J == 0 or parents.shape != (J,) or len(joint_names) != J:
        raise ValueError('Skeleton joints, parents and names must have the same nonzero length')
    if isinstance(skin, SparseSkin):
        if len(skin) != scene.vertex_count or skin.ids.shape != skin.weights.shape:
            raise ValueError(f'Sparse skin of {len(skin)} rows does not match {scene.vertex_count} vertices')
        if skin.ids.size and (skin.ids.min() < 0 or skin.ids.max() >= J):
            raise ValueError('Sparse skin joint index out of range')
    elif skin.shape != (scene.vertex_count, J):
        raise ValueError(f'Skin shape {skin.shape} does not match ({scene.vertex_count}, {J})')
    if np.any(parents >= J) or np.any(parents < -1):
        raise ValueError('Invalid joint parent index')
    shift = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    heads = joints + shift

    builder = Builder()
    data, nodes = builder.data, builder.data['nodes']

    # Armature wrapper + joints: joint j is node 1 + j.
    roots = [int(j) for j in np.flatnonzero(parents < 0)]
    if not roots:
        raise ValueError('Skeleton has no root joint')
    nodes.append({'name': armature_name, 'children': [1 + j for j in roots]})
    for j in range(J):
        p = int(parents[j])
        local = heads[j] - (heads[p] if p >= 0 else 0.)
        nodes.append({'name': str(joint_names[j]), 'translation': local.tolist(),
                      'children': [1 + int(c) for c in np.flatnonzero(parents == j)]})
    ibm = np.broadcast_to(np.eye(4), (J, 4, 4)).copy()
    ibm[:, :3, 3] = -heads
    data['skins'] = [{'name': armature_name, 'skeleton': 0, 'joints': list(range(1, J + 1)),
                      'inverseBindMatrices': builder.accessor(ibm.transpose(0, 2, 1).reshape(J, 16), 'MAT4')}]
    fallback = roots[0]
    if isinstance(skin, SparseSkin):
        k = skin.ids.shape[1]
        if group_per_vertex is not None and group_per_vertex > 0:
            k = min(k, group_per_vertex)
        ids_all, weights_all = _finish_sparse(skin.ids[:, :k], skin.weights[:, :k], fallback, normalize)
    else:
        ids_all, weights_all = _sparsify(skin, group_per_vertex, fallback, normalize)
    joint_component = np.uint8 if J <= 256 else np.uint16
    if J > 65536:
        raise ValueError('Too many joints for a glTF skin')
    integer_weights = normalize and weight_bits != 32
    weights_all = (_quantize_weights(weights_all) if integer_weights
                   else weights_all.astype(np.float32))

    scene_nodes = [0]
    meshes = []
    vertex_offset = 0
    src_meshes = document.data['meshes']
    for inst in scene.instances:
        m3 = inst.world[:3, :3]
        try:
            normal_m = np.linalg.inv(m3).T
        except np.linalg.LinAlgError:
            normal_m = m3
        mesh = {'primitives': []}
        if src_meshes[inst.mesh].get('name'):
            mesh['name'] = src_meshes[inst.mesh]['name']
        for prim in inst.primitives:
            n = len(prim.positions)
            src = prim.json
            attrs = {'POSITION': _write_attribute(builder, prim.positions + shift, bounds=True)}
            if 'NORMAL' in src['attributes']:
                normals = document.accessor(src['attributes']['NORMAL'])
                if len(normals) != n:
                    raise ValueError('Attribute NORMAL count does not match POSITION')
                attrs['NORMAL'] = _write_attribute(builder, _unit(normals.astype(np.float64) @ normal_m.T))
            sl = slice(vertex_offset, vertex_offset + n)
            for s in range(ids_all.shape[1] // 4):
                cols = slice(4 * s, 4 * s + 4)
                attrs[f'JOINTS_{s}'] = _write_attribute(builder, ids_all[sl, cols].astype(joint_component))
                attrs[f'WEIGHTS_{s}'] = _write_attribute(builder, weights_all[sl, cols],
                                                        normalized=integer_weights)
            vertex_offset += n
            out = {'attributes': attrs}
            indices = prim.faces.reshape(-1, 1).astype(np.uint16 if n <= 65535 else np.uint32)
            out['indices'] = _write_attribute(builder, indices, target=ELEMENT_ARRAY_BUFFER)
            mesh['primitives'].append(out)
        meshes.append(mesh)
        nodes.append({'name': inst.name, 'mesh': len(meshes) - 1, 'skin': 0})
        scene_nodes.append(len(nodes) - 1)
    if meshes:
        data['meshes'] = meshes
    data['scenes'] = [{'nodes': scene_nodes}]
    builder.write(filepath)


def asset_mesh_glb_bytes(vertices_yup: np.ndarray, faces: Optional[np.ndarray], vertex_bias, names) -> bytes:
    """Minimal GLB of raw mesh parts (no materials) for re-skinning an Asset."""
    builder = Builder()
    meshes, scene_nodes = [], []
    start = 0
    for i, end in enumerate(vertex_bias):
        verts = vertices_yup[start:end]
        prim = {'attributes': {'POSITION': _write_attribute(builder, verts, bounds=True)}}
        if faces is not None:
            part = faces[(faces >= start).all(axis=1) & (faces < end).all(axis=1)] - start
            if len(part):
                idx = part.reshape(-1, 1).astype(np.uint32)
                prim['indices'] = _write_attribute(builder, idx, target=ELEMENT_ARRAY_BUFFER)
            else:
                prim['mode'] = 0
        else:
            prim['mode'] = 0
        meshes.append({'primitives': [prim]})
        builder.data['nodes'].append({'name': str(names[i]), 'mesh': i})
        scene_nodes.append(i)
        start = end
    builder.data['meshes'] = meshes
    builder.data['scenes'] = [{'nodes': scene_nodes}]
    buf = io.BytesIO()
    builder.write(buf)
    return buf.getvalue()
