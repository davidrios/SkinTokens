"""Native glTF 2.0 reader: standard-library JSON/binary I/O, NumPy and SciPy only.

Adapted from AnyCross ``motion_lib/GLB.py`` (``Document``, ``_compose``,
``_decompose``). Never imports ``bpy``/``mathutils``/``pygltflib`` and never
falls back to Blender.
"""
from __future__ import annotations

import base64
import json
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote

import numpy as np
from scipy.spatial.transform import Rotation

from .limits import InputLimits, check_limit

_DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
# Resource bound for untrusted files. Accessor data with a bufferView is bounded
# by the file itself; a zero-initialized (sparse-only) accessor is not.
_MAX_ZERO_ACCESSOR_ELEMENTS = 2**26

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


@contextmanager
def malformed_as_value_error(name):
    """Report a structurally malformed glTF (missing key, wrong JSON type, bad
    index) as the ValueError every other invalid-file check raises."""
    try:
        yield
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'Malformed glTF {name}: {type(exc).__name__}: {exc}') from exc


def compose(t, q, s):
    """TRS (XYZW quaternion) -> 4x4 matrices, broadcasting over leading axes."""
    shape = np.broadcast_shapes(np.shape(t)[:-1], np.shape(q)[:-1], np.shape(s)[:-1])
    m = np.broadcast_to(np.eye(4), shape + (4, 4)).copy()
    m[..., :3, :3] = Rotation.from_quat(np.asarray(q, dtype=float).reshape(-1, 4)).as_matrix().reshape(shape + (3, 3)) * np.asarray(s)[..., None, :]
    m[..., :3, 3] = t
    return m


def decompose(m):
    """mathutils-compatible column normalization, including negative determinant.

    Returns ``(translation, xyzw_quaternion, scale)``. Sheared or singular
    matrices raise ``ValueError``.
    """
    translation = m[..., :3, 3].copy()
    a = np.array(m[..., :3, :3], copy=True)
    s = np.linalg.norm(a, axis=-2)
    s *= np.where(np.linalg.det(a) < 0, -1., 1.)[..., None]
    if np.any(np.abs(s) < 1e-15):
        raise ValueError('Singular node transform (zero scale) is not supported')
    a /= s[..., None, :]
    if not np.allclose(np.swapaxes(a, -1, -2) @ a, np.eye(3), atol=2e-5):
        raise ValueError('Sheared transform is not supported')
    q = Rotation.from_matrix(a.reshape(-1, 3, 3)).as_quat().reshape(a.shape[:-2] + (4,))
    # mathutils matrix.to_quaternion returns the hemisphere with nonnegative W.
    q *= np.where(q[..., 3:4] < 0, -1., 1.)
    return translation, q, s


def node_local_matrix(node: dict) -> np.ndarray:
    """Local 4x4 transform of a glTF node (``matrix`` or TRS)."""
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=float).reshape(4, 4).T
    return compose(np.asarray(node.get('translation', [0., 0., 0.]), dtype=float),
                   np.asarray(node.get('rotation', [0., 0., 0., 1.]), dtype=float),
                   np.asarray(node.get('scale', [1., 1., 1.]), dtype=float))


class Document:
    """Selective JSON/buffer reader. Accessors support stride, sparse and normalization.

    ``source`` is a ``.glb``/``.gltf`` path, or raw bytes (``name`` then supplies
    the extension; external resources are unavailable for in-memory input).
    With ``limits`` the input is untrusted: external buffer files are
    refused (images are never read), so it can never read other files from the server.
    """

    def __init__(self, source: Union[str, Path, bytes], name: Optional[str] = None,
                 limits: Optional[InputLimits] = None):
        self.limits = limits
        if isinstance(source, (bytes, bytearray)):
            self.path = Path(name or 'memory.glb')
            self.in_memory = True
            if limits is not None:
                check_limit(len(source), limits.max_file_bytes, 'file size (bytes)')
            raw = bytes(source)
        else:
            self.path = Path(source)
            self.in_memory = False
            if limits is not None:
                check_limit(self.path.stat().st_size, limits.max_file_bytes, 'file size (bytes)')
            raw = self.path.read_bytes()
        self.tail = b''
        self.binary = b''
        suffix = self.path.suffix.lower()
        if suffix == '.glb' or raw[:4] == b'glTF':
            if len(raw) < 20 or struct.unpack_from('<III', raw) != (GLB_MAGIC, 2, len(raw)):
                raise ValueError(f'Invalid GLB 2.0 header: {self.path}')
            length, kind = struct.unpack_from('<II', raw, 12)
            if kind != CHUNK_JSON or 20 + length > len(raw):
                raise ValueError('Invalid GLB JSON chunk')
            try:
                self.data = json.loads(raw[20:20 + length])
            except RecursionError:
                raise ValueError('Invalid GLB JSON chunk: nested too deeply') from None
            self.tail = raw[20 + length:]
            offset = 20 + length
            while offset < len(raw):
                if offset + 8 > len(raw):
                    raise ValueError('Truncated GLB chunk')
                size, kind = struct.unpack_from('<II', raw, offset)
                if offset + 8 + size > len(raw):
                    raise ValueError('Truncated GLB chunk payload')
                if kind == CHUNK_BIN:
                    self.binary = raw[offset + 8:offset + 8 + size]
                offset += 8 + size
        elif suffix == '.gltf':
            try:
                self.data = json.loads(raw)
            except RecursionError:
                raise ValueError('Invalid glTF JSON: nested too deeply') from None
        else:
            raise ValueError('GLB reader accepts .glb/.gltf only')
        if not isinstance(self.data, dict) or not isinstance(self.data.get('asset'), dict):
            raise ValueError('Invalid glTF JSON: expected an object with an asset')
        if self.data.get('asset', {}).get('version') != '2.0':
            raise ValueError('Only glTF 2.0 is supported')
        self._buffers = {}
        self._accessors = {}

    def _external(self, uri: str) -> bytes:
        if '://' in uri:
            raise ValueError('Remote glTF resources are not supported')
        if self.limits is not None:
            raise ValueError('An uploaded glTF must embed its buffers; '
                             'external files are not supported')
        if self.in_memory:
            raise ValueError('An in-memory glTF must embed its resources; external files are not supported')
        return (self.path.parent / unquote(uri)).read_bytes()

    def buffer(self, index):
        if index not in self._buffers:
            desc = self.data['buffers'][index]
            uri = desc.get('uri')
            if uri is None:
                value = self.binary
            elif not isinstance(uri, str):
                raise ValueError('glTF buffer uri must be a string')
            elif uri.startswith('data:'):
                header, value = uri.split(',', 1)
                value = base64.b64decode(value) if ';base64' in header else unquote(value).encode()
            else:
                value = self._external(uri)
            if len(value) < desc['byteLength']:
                raise ValueError('Truncated glTF buffer')
            self._buffers[index] = value
        return self._buffers[index]

    def view_bytes(self, view_index) -> bytes:
        view = self.data['bufferViews'][view_index]
        if 'EXT_meshopt_compression' in view.get('extensions', {}):
            raise ValueError('EXT_meshopt_compression buffer views are not supported')
        start = view.get('byteOffset', 0)
        data = self.buffer(view['buffer'])
        if start + view['byteLength'] > len(data):
            raise ValueError('Buffer view exceeds its buffer')
        return bytes(data[start:start + view['byteLength']])

    def _view_array(self, view_index, offset, count, width, dtype):
        view = self.data['bufferViews'][view_index]
        if 'EXT_meshopt_compression' in view.get('extensions', {}):
            raise ValueError('EXT_meshopt_compression accessors are not supported')
        dtype = np.dtype(dtype)
        stride = view.get('byteStride', width * dtype.itemsize)
        end = offset + (count - 1) * stride + width * dtype.itemsize if count else offset
        if offset < 0 or end > view['byteLength'] or stride < width * dtype.itemsize:
            raise ValueError('Accessor exceeds its buffer view')
        return np.ndarray((count, width), dtype=dtype, buffer=self.buffer(view['buffer']),
                          offset=view.get('byteOffset', 0) + offset,
                          strides=(stride, dtype.itemsize)).copy()

    def accessor(self, index):
        """Decoded accessor as a ``(count, width)`` array (normalized ints -> float)."""
        if index in self._accessors:
            return self._accessors[index]
        a = self.data['accessors'][index]
        dtype, width, count = _DTYPES[a['componentType']], _WIDTHS[a['type']], a['count']
        if a['type'] in ('MAT2', 'MAT3') and np.dtype(dtype).itemsize < 4:
            raise ValueError('Padded integer matrix accessors are not supported')
        if 'bufferView' in a:
            result = self._view_array(a['bufferView'], a.get('byteOffset', 0), count, width, dtype)
        else:
            if count * width > _MAX_ZERO_ACCESSOR_ELEMENTS:
                raise ValueError(f'Accessor without a bufferView is too large: {count} x {a["type"]}')
            result = np.zeros((count, width), dtype=dtype)
        sparse = a.get('sparse')
        if sparse:
            ids, vals = sparse['indices'], sparse['values']
            indices = self._view_array(ids['bufferView'], ids.get('byteOffset', 0), sparse['count'], 1, _DTYPES[ids['componentType']])[:, 0]
            result[indices] = self._view_array(vals['bufferView'], vals.get('byteOffset', 0), sparse['count'], width, dtype)
        if a.get('normalized') and np.issubdtype(result.dtype, np.integer):
            info = np.iinfo(result.dtype)
            result = np.maximum(result.astype(float) / info.max, -1.)
        self._accessors[index] = result
        return result
