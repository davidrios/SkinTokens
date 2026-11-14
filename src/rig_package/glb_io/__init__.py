"""Native glTF/GLB I/O (NumPy/SciPy only; no bpy, no FBX).

Reader/writer core adapted from AnyCross ``motion_lib`` (see ``GLB_IO.md``
there); mesh/skin handling is specific to SkinTokens.
"""
from .builder import Builder, write_glb_bytes
from .document import Document, compose, decompose
from .limits import InputLimits
from .scene import (
    MAX_INFLUENCES,
    SUPPORTED_SCENE_EXT,
    Y_UP_TO_Z_UP,
    Z_UP_TO_Y_UP,
    MeshInstance,
    Scene,
    SkinnedRig,
    SparseSkin,
    export_skinned_glb,
    load_scene,
    prepare_skinned_rig,
    write_skinned_glb,
)
from .rig_payload import (RigPayload, apply_rig, build_glb, decode_rig_payload, encode_rig_payload,
                          scene_layout)

__all__ = [
    'Builder', 'write_glb_bytes', 'Document', 'compose', 'decompose', 'InputLimits',
    'SUPPORTED_SCENE_EXT', 'Y_UP_TO_Z_UP', 'Z_UP_TO_Y_UP', 'MeshInstance',
    'Scene', 'SkinnedRig', 'SparseSkin', 'export_skinned_glb', 'load_scene',
    'prepare_skinned_rig', 'write_skinned_glb', 'MAX_INFLUENCES',
    'RigPayload', 'apply_rig', 'build_glb', 'decode_rig_payload', 'encode_rig_payload', 'scene_layout',
]
