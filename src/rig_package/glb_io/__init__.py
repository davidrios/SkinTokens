"""Native glTF/GLB I/O (NumPy/SciPy only; no bpy, no FBX).

Reader/writer core adapted from AnyCross ``motion_lib`` (see ``GLB_IO.md``
there); mesh/skin handling is specific to SkinTokens.
"""
from .builder import Builder, write_glb_bytes
from .document import Document, compose, decompose
from .limits import InputLimits
from .scene import (
    SUPPORTED_SCENE_EXT,
    Y_UP_TO_Z_UP,
    Z_UP_TO_Y_UP,
    MeshInstance,
    Scene,
    SparseSkin,
    export_skinned_glb,
    load_scene,
)

__all__ = [
    'Builder', 'write_glb_bytes', 'Document', 'compose', 'decompose', 'InputLimits',
    'SUPPORTED_SCENE_EXT', 'Y_UP_TO_Z_UP', 'Z_UP_TO_Y_UP', 'MeshInstance',
    'Scene', 'SparseSkin', 'export_skinned_glb', 'load_scene',
]
