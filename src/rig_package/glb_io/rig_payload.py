"""Rig payloads: a skinned GLB minus the geometry the client already has.

A rigged GLB is mostly the input's own positions, normals and triangles; only
the skeleton and the per-vertex joint influences are new. A rig payload carries
just those (the :class:`~.scene.SkinnedRig` :func:`~.scene.write_skinned_glb`
combines with the scene), so the client of a server can rig the file it uploaded:
:func:`apply_rig` adds the rig to that file, keeping its materials, textures
and structure; :func:`build_glb` gives the very bytes the server would have
sent as a GLB (the skinned mesh only). A three.js client attaches the skin to
the loaded scene instead (``clients/threejs/skintokens-rig.js``).

Format (version 1, little-endian), the chunk layout of GLB::

    magic   b"SKRG"
    u32     version (1)
    u32     total length in bytes
    chunk   u32 length, u32 type 0x4E4F534A ("JSON"), UTF-8 JSON padded with spaces
    chunk   u32 length, u32 type 0x004E4942 ("BIN\\0"), the arrays (zlib-compressed
            by default), padded with zeros

The JSON::

    {"format": "skintokens-rig", "version": 1,
     "armature_name": str, "joint_names": [J str], "parents": [J int, -1 for roots],
     "joint_heads": [J [x, y, z]],     # bind pose, glTF world space (Y-up), before
     "ground_offset": [x, y, z],       # this shift of meshes and joints together
     "vertex_count": N, "influences": K,             # K a multiple of 4 (4)
     "joint_indices": {"byte_offset", "byte_length", "component_type": "uint8"|"uint16"},
     "joint_weights": {"byte_offset", "byte_length", "component_type": "uint16"|"float32",
                       "normalized": bool},          # uint16 rows sum to 65535
     "bin": {"compression": "zlib"|"none", "byte_length": raw, "stored_byte_length": stored},
     "layout": [{"node": int, "mesh": int, "name": str, "skinned_input": bool,
                 "primitives": [{"index": int, "vertex_count": int}]}]}

``joint_indices``/``joint_weights`` are ``(N, K)`` row-major arrays at their
offsets in the (decompressed) BIN data, row ``i`` for vertex ``i`` in
``layout`` order: the default scene's mesh nodes depth-first, each node's
triangle primitives (``index`` in the glTF mesh; points, lines and primitives
without faces are left out) in order, the primitive's vertices in accessor
order. A mesh used by several nodes appears once per node. ``skinned_input``
nodes had a skin in the input, so their vertices are used as stored, not moved
by the node transform.
"""
from __future__ import annotations

import io
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .document import CHUNK_BIN, CHUNK_JSON, GLB_MAGIC, malformed_as_value_error
from .scene import MAX_INFLUENCES, Scene, SkinnedRig, load_scene, write_skinned_glb

MAGIC = b'SKRG'
VERSION = 1
FORMAT = 'skintokens-rig'
# Bound on a payload's decompressed arrays, whatever it declares.
MAX_BIN_BYTES = 1 << 31

_INDEX_TYPES = {'uint8': np.dtype('<u1'), 'uint16': np.dtype('<u2')}
_WEIGHT_TYPES = {'uint16': np.dtype('<u2'), 'float32': np.dtype('<f4')}


@dataclass(frozen=True)
class RigPayload:
    """A decoded rig payload: the rig and the vertex layout it was made for."""
    rig: SkinnedRig
    layout: list
    vertex_count: int


def scene_layout(scene: Scene) -> list:
    """The vertex order of ``scene`` (the order of ``scene.vertices()``) as the
    ``layout`` of a rig payload."""
    return [{'node': int(inst.node), 'mesh': int(inst.mesh), 'name': inst.name,
             'skinned_input': inst.skin is not None,
             'primitives': [{'index': int(p.index), 'vertex_count': int(len(p.positions))}
                            for p in inst.primitives]}
            for inst in scene.instances]


def _type_name(dtype, table) -> str:
    for name, t in table.items():
        if np.dtype(dtype) == t:
            return name
    raise ValueError(f'unsupported array type {dtype}')


def encode_rig_payload(scene: Scene, rig: SkinnedRig, *, compress: bool = True, level: int = 6) -> bytes:
    """``rig`` (computed for ``scene``) as a rig payload."""
    rig.validate(scene.vertex_count)
    ids = np.ascontiguousarray(rig.ids, dtype=rig.ids.dtype.newbyteorder('<'))
    weights = np.ascontiguousarray(rig.weights, dtype=rig.weights.dtype.newbyteorder('<'))
    ids_bytes = ids.tobytes()
    weights_offset = len(ids_bytes) + (-len(ids_bytes) % 4)
    raw = ids_bytes + bytes(weights_offset - len(ids_bytes)) + weights.tobytes()
    stored = zlib.compress(raw, level) if compress else raw
    header = {
        'format': FORMAT,
        'version': VERSION,
        'armature_name': rig.armature_name,
        'joint_names': list(rig.joint_names),
        'parents': [int(p) for p in rig.parents],
        # Python floats: JSON keeps float64 values exactly.
        'joint_heads': np.asarray(rig.joints, dtype=np.float64).tolist(),
        'ground_offset': np.asarray(rig.offset, dtype=np.float64).tolist(),
        'vertex_count': int(len(ids)),
        'influences': int(ids.shape[1]),
        'joint_indices': {'byte_offset': 0, 'byte_length': len(ids_bytes),
                          'component_type': _type_name(ids.dtype, _INDEX_TYPES)},
        'joint_weights': {'byte_offset': weights_offset, 'byte_length': weights.nbytes,
                          'component_type': _type_name(weights.dtype, _WEIGHT_TYPES),
                          'normalized': rig.normalized_weights},
        'bin': {'compression': 'zlib' if compress else 'none',
                'byte_length': len(raw), 'stored_byte_length': len(stored)},
        'layout': scene_layout(scene),
    }
    text = json.dumps(header, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    text += b' ' * (-len(text) % 4)
    stored += bytes(-len(stored) % 4)
    length = 12 + 8 + len(text) + 8 + len(stored)
    return b''.join([MAGIC, struct.pack('<II', VERSION, length),
                     struct.pack('<II', len(text), CHUNK_JSON), text,
                     struct.pack('<II', len(stored), CHUNK_BIN), stored])


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(f'invalid rig payload: {message}')


def _int(value, name: str, minimum: int = 0, maximum: int = MAX_BIN_BYTES) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum,
             f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _array(raw: bytes, spec, table, name: str, shape) -> np.ndarray:
    _require(isinstance(spec, dict), f'{name} must be an object')
    type_name = spec.get('component_type')
    _require(type_name in table, f'{name}.component_type must be one of {sorted(table)}')
    dtype = table[type_name]
    offset = _int(spec.get('byte_offset'), f'{name}.byte_offset')
    length = _int(spec.get('byte_length'), f'{name}.byte_length')
    _require(length == shape[0] * shape[1] * dtype.itemsize,
             f'{name}.byte_length does not match vertex_count x influences')
    _require(offset % dtype.itemsize == 0 and offset + length <= len(raw), f'{name} is out of bounds')
    return np.frombuffer(raw, dtype=dtype, count=shape[0] * shape[1], offset=offset).reshape(shape)


def _chunks(data: bytes):
    _require(len(data) >= 12 and data[:4] == MAGIC, 'not a rig payload (bad magic)')
    version, length = struct.unpack_from('<II', data, 4)
    _require(version == VERSION, f'unsupported version {version} (expected {VERSION})')
    _require(length == len(data), f'declared length {length} does not match {len(data)} bytes')
    chunks, pos = [], 12
    while pos < len(data):
        _require(pos + 8 <= len(data), 'truncated chunk header')
        size, kind = struct.unpack_from('<II', data, pos)
        _require(pos + 8 + size <= len(data), 'truncated chunk')
        chunks.append((kind, data[pos + 8:pos + 8 + size]))
        pos += 8 + size
    _require(len(chunks) == 2 and chunks[0][0] == CHUNK_JSON and chunks[1][0] == CHUNK_BIN,
             'expected a JSON chunk then a BIN chunk')
    return chunks[0][1], chunks[1][1]


def decode_rig_payload(data: bytes) -> RigPayload:
    """Parse and check a rig payload (``ValueError`` if it is malformed)."""
    data = bytes(data)
    text, stored = _chunks(data)
    try:
        header = json.loads(text.decode('utf-8'))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValueError('invalid rig payload: malformed JSON chunk') from None
    _require(isinstance(header, dict) and header.get('format') == FORMAT, f'format is not {FORMAT!r}')

    bin_info = header.get('bin')
    _require(isinstance(bin_info, dict), 'bin must be an object')
    raw_length = _int(bin_info.get('byte_length'), 'bin.byte_length')
    stored_length = _int(bin_info.get('stored_byte_length'), 'bin.stored_byte_length')
    _require(stored_length <= len(stored), 'BIN chunk shorter than bin.stored_byte_length')
    stored = stored[:stored_length]
    compression = bin_info.get('compression')
    if compression == 'zlib':
        inflater = zlib.decompressobj()
        try:
            # max_length bounds the output, whatever the stream expands to.
            raw = inflater.decompress(stored, raw_length)
        except zlib.error as exc:
            raise ValueError(f'invalid rig payload: corrupt zlib data ({exc})') from None
        _require(inflater.eof and not inflater.unconsumed_tail and not inflater.unused_data,
                 'zlib data does not match bin.byte_length')
    elif compression == 'none':
        raw = stored
    else:
        raise ValueError(f'invalid rig payload: unsupported compression {compression!r}')
    _require(len(raw) == raw_length, 'BIN data does not match bin.byte_length')

    names = header.get('joint_names')
    parents = header.get('parents')
    heads = header.get('joint_heads')
    _require(isinstance(names, list) and all(isinstance(n, str) for n in names), 'joint_names must be strings')
    J = len(names)
    _require(isinstance(parents, list) and len(parents) == J, 'parents must have one entry per joint')
    parents = [_int(p, 'parents[]', -1, J - 1) for p in parents]
    _require(isinstance(heads, list) and len(heads) == J, 'joint_heads must have one entry per joint')
    offset = header.get('ground_offset')
    try:
        heads = np.array(heads, dtype=np.float64).reshape(J, 3) if J else np.zeros((0, 3))
        offset = np.array(offset, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        raise ValueError('invalid rig payload: joint_heads and ground_offset must be 3-vectors') from None
    armature_name = header.get('armature_name')
    _require(isinstance(armature_name, str), 'armature_name must be a string')

    N = _int(header.get('vertex_count'), 'vertex_count')
    K = _int(header.get('influences'), 'influences', 4, MAX_INFLUENCES)
    ids = _array(raw, header.get('joint_indices'), _INDEX_TYPES, 'joint_indices', (N, K))
    weight_spec = header.get('joint_weights')
    weights = _array(raw, weight_spec, _WEIGHT_TYPES, 'joint_weights', (N, K))
    _require(weight_spec.get('normalized') is bool(weights.dtype == np.uint16),
             'joint_weights.normalized must be true for uint16 and false for float32')

    layout = header.get('layout')
    _require(isinstance(layout, list), 'layout must be a list')
    rig = SkinnedRig(armature_name=armature_name, joint_names=tuple(names),
                     parents=np.array(parents, dtype=np.int64), joints=heads, offset=offset,
                     ids=ids.astype(ids.dtype.newbyteorder('=')),
                     weights=weights.astype(weights.dtype.newbyteorder('=')))
    try:
        rig.validate(N)
    except ValueError as exc:
        raise ValueError(f'invalid rig payload: {exc}') from None
    return RigPayload(rig=rig, layout=layout, vertex_count=N)


def build_glb(source, payload, *, name: Optional[str] = None) -> bytes:
    """The skinned GLB for ``source`` and ``payload``: the same bytes the server
    returns as a GLB for that input.

    ``source`` is the input as uploaded (its bytes, with ``name`` giving its
    extension, or a path) or its :class:`Scene`; ``payload`` the payload's
    bytes or a :class:`RigPayload`. Raises ``ValueError`` when the payload was
    not made for this input.
    """
    scene, payload = _load_checked(source, payload, name)
    out = io.BytesIO()
    write_skinned_glb(scene, out, payload.rig)
    return out.getvalue()


def _load_checked(source, payload, name):
    if not isinstance(payload, RigPayload):
        payload = decode_rig_payload(payload)
    scene = source if isinstance(source, Scene) else load_scene(source, name=name)
    if payload.layout != scene_layout(scene):
        raise ValueError('rig payload does not match this input (its vertex layout differs: '
                         'apply it to the exact file that was uploaded)')
    return scene, payload


class _BinWriter:
    """Appends accessors to buffer 0 of a glTF JSON (the GLB's BIN chunk)."""

    def __init__(self, data: dict, blob: bytes):
        self.data = data
        self.blob = bytearray(blob)

    def accessor(self, values: np.ndarray, kind: str, component: int, *, normalized=False, target=None) -> int:
        self.blob.extend(bytes(-len(self.blob) % 4))
        raw = np.ascontiguousarray(values).tobytes()
        view = {'buffer': 0, 'byteOffset': len(self.blob), 'byteLength': len(raw)}
        if target is not None:
            view['target'] = target
        self.blob.extend(raw)
        views = self.data.setdefault('bufferViews', [])
        views.append(view)
        accessor = {'bufferView': len(views) - 1, 'componentType': component,
                    'count': int(len(values)), 'type': kind}
        if normalized:
            accessor['normalized'] = True
        accessors = self.data.setdefault('accessors', [])
        accessors.append(accessor)
        return len(accessors) - 1


def _bin_buffer(document, data: dict) -> bytes:
    """Make buffer 0 the GLB's BIN chunk, returning its current bytes.

    A GLB's BIN chunk is buffer 0 already. Otherwise (a .gltf, or a GLB
    without one) a new buffer 0 is made of the input's data-URI buffers, which
    a GLB would otherwise carry as base64 text: their bytes are unchanged, their
    buffer views only point into the new buffer. Buffers in external files
    stay where they are.
    """
    buffers = data.setdefault('buffers', [])
    if buffers and 'uri' not in buffers[0]:
        return document.binary
    used = set(data.get('extensionsUsed', [])) | set(data.get('extensionsRequired', []))
    if buffers and 'EXT_meshopt_compression' in used:
        raise ValueError('EXT_meshopt_compression files are not supported')
    blob = bytearray()
    kept = [{'byteLength': 0}]
    moved = {}  # input buffer -> (new buffer, byte offset in it)
    for i, desc in enumerate(buffers):
        uri = desc.get('uri')
        if isinstance(uri, str) and uri.startswith('data:'):
            blob.extend(bytes(-len(blob) % 8))  # keeps every accessor aligned
            moved[i] = (0, len(blob))
            blob.extend(document.buffer(i)[:desc['byteLength']])
        else:
            kept.append(desc)
            moved[i] = (len(kept) - 1, 0)
    for view in data.get('bufferViews', []):
        index, offset = moved[view['buffer']]
        view['buffer'] = index
        if offset:
            view['byteOffset'] = view.get('byteOffset', 0) + offset
    data['buffers'] = kept
    return bytes(blob)


def apply_rig(source, payload, *, name: Optional[str] = None, auto_ground: bool = False) -> bytes:
    """The input file with the payload's rig added, as a GLB.

    Unlike :func:`build_glb` (the server's GLB: the skinned mesh only), this
    keeps everything the input has (node hierarchy and transforms, materials,
    textures, UVs, animations, extensions, extras) and only adds a skeleton
    (an armature node under the default scene, one node per joint), skins, and
    ``JOINTS_n``/``WEIGHTS_n`` on the rigged meshes' primitives (replacing any
    the input had; skins no node refers to any more are dropped). Vertex data is
    not changed: each skin's inverse bind
    matrices take in the mesh node's world transform, which glTF ignores for a
    skinned mesh. A mesh used by several nodes is copied (its geometry
    accessors shared) so each use carries its own weights. Primitives the rig
    has no rows for (points, lines, no triangles) take their nearest rigged
    vertex's influences, as glTF needs them on every primitive of a skinned
    mesh. The model keeps its place unless ``auto_ground`` is true, which applies
    the server's ground offset (the scene's bottom on Z=0) by translating the
    scene's root nodes and the armature together; vertex data is unchanged.

    ``source`` and ``payload`` as for :func:`build_glb`.
    """
    scene, payload = _load_checked(source, payload, name)
    document = scene.document
    with malformed_as_value_error(document.path.name):
        return _apply_rig(scene, document, payload.rig, auto_ground)


def _translate_node(node: dict, shift: np.ndarray) -> None:
    """Add ``shift`` to a node's local transform (a world-space shift for a root)."""
    if 'matrix' in node:
        matrix = list(node['matrix'])
        for k in range(3):
            matrix[12 + k] += float(shift[k])
        node['matrix'] = matrix
    else:
        translation = node.get('translation', [0., 0., 0.])
        node['translation'] = [float(translation[k]) + float(shift[k]) for k in range(3)]


def _apply_rig(scene, document, rig: SkinnedRig, auto_ground: bool = False) -> bytes:
    import copy

    from .builder import ARRAY_BUFFER
    shift = np.asarray(rig.offset, dtype=np.float64) if auto_ground else None

    data = copy.deepcopy(document.data)
    writer = _BinWriter(data, _bin_buffer(document, data))
    nodes = data.setdefault('nodes', [])
    J = len(rig.joints)
    heads = np.asarray(rig.joints, dtype=np.float64)

    # Skeleton: an armature node at the scene root (identity, so the joint
    # heads are in the scene's own space), then joint j at node base + 1 + j.
    base = len(nodes)
    roots = [int(j) for j in np.flatnonzero(rig.parents < 0)]
    nodes.append({'name': rig.armature_name, 'children': [base + 1 + j for j in roots]})
    for j in range(J):
        p = int(rig.parents[j])
        node = {'name': rig.joint_names[j],
                'translation': (heads[j] - (heads[p] if p >= 0 else 0.)).tolist()}
        children = [base + 1 + int(c) for c in np.flatnonzero(rig.parents == j)]
        if children:
            node['children'] = children
        nodes.append(node)
    scenes = data.get('scenes')
    if not scenes:
        scenes = data['scenes'] = [{'nodes': []}]
    scene_nodes = scenes[data.get('scene', 0)].setdefault('nodes', [])
    if shift is not None and np.any(shift):
        # Move the whole scene (and the armature with it) by translating its
        # root nodes. Skinned vertices follow the armature, so the vertex data
        # and the inverse bind matrices stay as they are.
        if scene_nodes:
            moved_roots = list(scene_nodes)
        else:  # no root list: readers (and ours) take every parentless node
            children = {c for n in nodes[:base] for c in n.get('children', [])}
            moved_roots = [i for i in range(base) if i not in children]
        for i in moved_roots:
            _translate_node(nodes[i], shift)
        nodes[base]['translation'] = shift.tolist()
    scene_nodes.append(base)
    joint_nodes = list(range(base + 1, base + 1 + J))

    skins = data.setdefault('skins', [])
    skin_by_world: dict = {}

    def skin_for(world: np.ndarray) -> int:
        key = world.tobytes()
        if key not in skin_by_world:
            ibm = np.broadcast_to(np.eye(4), (J, 4, 4)).copy()
            ibm[:, :3, 3] = -heads
            ibm = ibm @ world  # joint bind pose^-1 x the mesh's placement
            skins.append({'name': rig.armature_name, 'skeleton': base, 'joints': joint_nodes,
                          'inverseBindMatrices': writer.accessor(
                              ibm.transpose(0, 2, 1).reshape(J, 16).astype('<f4'), 'MAT4', 5126)})
            skin_by_world[key] = len(skins) - 1
        return skin_by_world[key]

    joint_component = 5121 if rig.ids.dtype == np.uint8 else 5123
    weight_component = 5123 if rig.normalized_weights else 5126
    influence_sets = rig.ids.shape[1] // 4
    used_meshes = set()
    row = 0
    for inst in scene.instances:
        node = nodes[inst.node]
        mesh_index = inst.mesh
        if mesh_index in used_meshes:
            data['meshes'].append(copy.deepcopy(data['meshes'][inst.mesh]))
            mesh_index = len(data['meshes']) - 1
            node['mesh'] = mesh_index
        used_meshes.add(inst.mesh)
        node['skin'] = skin_for(inst.world)

        # This node's rows, per primitive index, and its rigged vertices.
        rows = {}
        for prim in inst.primitives:
            n = len(prim.positions)
            rows[prim.index] = slice(row, row + n)
            row += n
        rigged = np.concatenate([p.positions for p in inst.primitives])
        node_rows = np.concatenate([np.arange(s.start, s.stop) for s in rows.values()])
        tree = None

        for index, prim in enumerate(data['meshes'][mesh_index]['primitives']):
            attrs = prim['attributes']
            if index in rows:
                take = rows[index]
            else:
                if 'POSITION' not in attrs:
                    continue
                local = document.accessor(attrs['POSITION']).astype(np.float64)
                world = local @ inst.world[:3, :3].T + inst.world[:3, 3]
                if tree is None:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(rigged)
                take = node_rows[tree.query(world, k=1)[1]] if len(world) else np.zeros(0, int)
            ids, weights = rig.ids[take], rig.weights[take]
            if len(ids) == 0:
                continue  # no vertices to skin (glTF has no empty accessors)
            for key in [k for k in attrs if k.startswith(('JOINTS_', 'WEIGHTS_'))]:
                del attrs[key]
            for s in range(influence_sets):
                cols = slice(4 * s, 4 * s + 4)
                attrs[f'JOINTS_{s}'] = writer.accessor(
                    ids[:, cols].astype(ids.dtype.newbyteorder('<')), 'VEC4', joint_component,
                    target=ARRAY_BUFFER)
                attrs[f'WEIGHTS_{s}'] = writer.accessor(
                    weights[:, cols].astype(weights.dtype.newbyteorder('<')), 'VEC4', weight_component,
                    normalized=rig.normalized_weights, target=ARRAY_BUFFER)

    # Skins the input had are unreferenced now that its mesh nodes point at
    # ours: glTF readers warn about (and some reject) unused skins.
    used = sorted({node['skin'] for node in nodes if 'skin' in node})
    if len(used) != len(skins):
        remap = {old: new for new, old in enumerate(used)}
        data['skins'] = [skins[old] for old in used]
        for node in nodes:
            if 'skin' in node:
                node['skin'] = remap[node['skin']]

    data['buffers'][0]['byteLength'] = len(writer.blob)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    text += b' ' * (-len(text) % 4)
    blob = bytes(writer.blob) + bytes(-len(writer.blob) % 4)
    length = 12 + 8 + len(text) + 8 + len(blob)
    return b''.join([struct.pack('<III', GLB_MAGIC, 2, length),
                     struct.pack('<II', len(text), CHUNK_JSON), text,
                     struct.pack('<II', len(blob), CHUNK_BIN), blob])
