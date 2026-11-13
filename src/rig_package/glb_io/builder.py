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
        self.blob.extend(payload)
        return offset

    def view(self, payload, target=None):
        offset = self.append(payload)
        views = self.data.setdefault('bufferViews', [])
        entry = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(payload)}
        if target is not None:
            entry['target'] = target
        views.append(entry)
        return len(views) - 1

    def accessor(self, values, kind, *, component=5126, bounds=False, normalized=False, target=None):
        values = np.asarray(values, dtype=_COMPONENT_DTYPES[component])
        if not np.all(np.isfinite(values)):
            raise ValueError('Cannot export non-finite accessor values')
        entry = {'bufferView': self.view(values.tobytes(), target=target), 'componentType': component,
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
        chunks = struct.pack('<II', len(text), CHUNK_JSON) + text
        if self.blob:
            binary = bytes(self.blob) + b'\0' * (-len(self.blob) % 4)
            chunks += struct.pack('<II', len(binary), CHUNK_BIN) + binary
        return struct.pack('<III', GLB_MAGIC, 2, len(chunks) + 12) + chunks

    def write(self, filepath):
        """Write the GLB to a ``.glb`` path or a writable binary stream."""
        if not hasattr(filepath, 'write') and Path(filepath).suffix.lower() != '.glb':
            raise ValueError('Native GLB export requires a .glb output path')
        write_glb_bytes(filepath, self.to_bytes())
