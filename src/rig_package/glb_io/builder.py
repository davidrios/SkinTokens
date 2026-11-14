"""Native GLB writer: packs JSON + one binary chunk with NumPy only.

Adapted from AnyCross ``motion_lib/glb_writer.py`` (``Builder``,
``write_glb_bytes``). There is no bpy exporter or fallback.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .document import CHUNK_BIN, CHUNK_JSON, GLB_MAGIC

_COMPONENT_DTYPES = {5126: '<f4', 5121: 'u1', 5123: '<u2', 5125: '<u4'}
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def write_glb_bytes(target, data):
    """Write finished GLB *data* to a path, or to a writable binary stream."""
    if hasattr(target, 'write'):
        target.write(data)
        return
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


class Builder:
    """Accumulate glTF JSON plus a single binary buffer, then write a GLB."""

    def __init__(self, generator: str = 'SkinTokens native GLB'):
        self.data = {
            'asset': {'version': '2.0', 'generator': generator},
            'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [],
        }
        self.blob = bytearray()

    def append(self, payload):
        self.blob.extend(b'\0' * (-len(self.blob) % 4))
        offset = len(self.blob)
        self.blob += payload  # any bytes-like: an array's memory is copied once, here
        return offset

    def view(self, payload, target=None):
        offset = self.append(payload)
        views = self.data.setdefault('bufferViews', [])
        entry = {'buffer': 0, 'byteOffset': offset, 'byteLength': memoryview(payload).nbytes}
        if target is not None:
            entry['target'] = target
        views.append(entry)
        return len(views) - 1

    def accessor(self, values, kind, *, component=5126, bounds=False, normalized=False, target=None):
        values = np.asarray(values, dtype=_COMPONENT_DTYPES[component])
        if not np.all(np.isfinite(values)):
            raise ValueError('Cannot export non-finite accessor values')
        # The array's own memory goes straight into the blob (no tobytes() copy).
        raw = memoryview(np.ascontiguousarray(values).reshape(-1).view(np.uint8))
        entry = {'bufferView': self.view(raw, target=target), 'componentType': component,
                 'count': len(values), 'type': kind}
        if normalized:
            entry['normalized'] = True
        if bounds and len(values):
            rows = values.reshape(len(values), -1)
            entry.update(min=rows.min(axis=0).tolist(), max=rows.max(axis=0).tolist())
        self.data.setdefault('accessors', []).append(entry)
        return len(self.data['accessors']) - 1

    def to_bytes(self) -> bytes:
        for node in self.data.get('nodes', []):
            if not node.get('children'):
                node.pop('children', None)
        if self.blob:
            self.data['buffers'] = [{'byteLength': len(self.blob)}]
        else:
            self.data.pop('buffers', None)
        text = json.dumps(self.data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf8')
        text += b' ' * (-len(text) % 4)
        json_chunk = struct.pack('<II', len(text), CHUNK_JSON)
        parts = [json_chunk, text]
        length = 12 + len(json_chunk) + len(text)
        if self.blob:
            pad = bytes(1) * (-len(self.blob) % 4)
            bin_size = len(self.blob) + len(pad)
            parts += [struct.pack('<II', bin_size, CHUNK_BIN), self.blob, pad]
            length += 8 + bin_size
        # One join: the binary buffer is copied once, into the final bytes.
        return b''.join([struct.pack('<III', GLB_MAGIC, 2, length)] + parts)

    def write(self, filepath):
        """Write the GLB to a ``.glb`` path or a writable binary stream."""
        if not hasattr(filepath, 'write') and Path(filepath).suffix.lower() != '.glb':
            raise ValueError('Native GLB export requires a .glb output path')
        write_glb_bytes(filepath, self.to_bytes())
