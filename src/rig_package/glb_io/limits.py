"""Resource limits for untrusted (uploaded) geometry.

Passing an :class:`InputLimits` to the loaders marks the input as untrusted:
external file references are refused (the file may only use embedded
resources), and sizes are checked from glTF metadata *before* large arrays are
decoded or allocated. Trusted local use (``limits=None``) is unrestricted.

Every limit can be overridden with an environment variable, see
:meth:`InputLimits.from_env`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields


def upload_limit_mb() -> int:
    """Shared upload and parser file-size limit, in MiB (1..1024)."""
    name = 'SKINTOKENS_MAX_UPLOAD_MB'
    raw = os.environ.get(name)
    try:
        value = 64 if raw is None else int(raw)
    except ValueError:
        logging.getLogger(__name__).warning('Invalid %s=%r; using default 64', name, raw)
        return 64
    clamped = min(max(value, 1), 1024)
    if value != clamped:
        logging.getLogger(__name__).warning('Clamped %s=%d to %d (allowed 1..1024)', name, value, clamped)
    return clamped


@dataclass(frozen=True)
class InputLimits:
    # size of the uploaded file
    max_file_bytes: int = 64 * 1024 * 1024
    # total mesh vertices / triangles over all mesh instances
    max_vertices: int = 2_000_000
    max_faces: int = 4_000_000
    # all nodes (each costs a transform in the hierarchy walk, mesh or not).
    # Rigged characters have a few hundred nodes at most; a few thousand
    # bounds the walk to well under a second while leaving ~50x headroom.
    max_nodes: int = 4096
    # mesh nodes (instancing one mesh many times multiplies its vertices)
    max_mesh_instances: int = 4096
    # primitives summed over mesh instances (bounds per-primitive work)
    max_primitives: int = 65536

    @classmethod
    def from_env(cls, prefix: str = 'SKINTOKENS_INPUT_') -> 'InputLimits':
        """File size uses SKINTOKENS_MAX_UPLOAD_MB; geometry uses ``prefix``."""
        values = {'max_file_bytes': upload_limit_mb() * 1024 * 1024}
        for f in fields(cls):
            if f.name == 'max_file_bytes':
                continue
            raw = os.environ.get(prefix + f.name.upper())
            if raw is not None and raw.strip():
                value = int(raw)
                if value <= 0:
                    raise ValueError(f'{prefix + f.name.upper()} must be positive')
                values[f.name] = value
        return cls(**values)


def check_limit(value: int, limit: int, what: str):
    if value > limit:
        raise ValueError(f'Input {what} exceeds the limit: {value} > {limit}')
