"""Native GLB/GLTF parser and rigged-GLB exporter (no bpy, no FBX).

Assets use Blender's Z-up convention, which the model was trained on: glTF
Y-up coordinates are converted with ``(x, y, z) -> (x, -z, y)`` on load and back
on export, exactly like Blender's importers/exporters. Geometry is expressed in
world space (skinned meshes in bind space), so ``matrix_world`` is identity.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import trimesh
from numpy import ndarray

from ..glb_io import Y_UP_TO_Z_UP, Z_UP_TO_Y_UP, Document, InputLimits, Scene, export_skinned_glb, load_scene
from ..glb_io.scene import asset_mesh_glb_bytes
from ..info.asset import Asset
from .abstract import AbstractParser


def _to_z_up(points: ndarray) -> ndarray:
    return np.asarray(points, dtype=np.float64) @ Y_UP_TO_Z_UP.T


def _to_y_up(points: ndarray) -> ndarray:
    return np.asarray(points, dtype=np.float64) @ Z_UP_TO_Y_UP.T


def _scene_mesh(scene: Scene, include_faces: bool = True, include_normals: bool = True):
    names, vertices, faces, vertex_bias, face_bias = [], [], [], [], []
    v_total, f_total = 0, 0
    for name, verts, part_faces in scene.parts():
        names.append(name)
        vertices.append(_to_z_up(verts).astype(np.float32))
        if include_faces:
            faces.append(part_faces.astype(np.int32) + v_total)
            f_total += len(part_faces)
            face_bias.append(f_total)
        v_total += len(verts)
        vertex_bias.append(v_total)
    vertices = np.concatenate(vertices, axis=0) if vertices else np.zeros((0, 3), dtype=np.float32)
    all_faces = np.concatenate(faces, axis=0) if include_faces and faces else None
    if all_faces is not None and len(all_faces) == 0:
        all_faces = None
    vertex_normals = face_normals = None
    if all_faces is not None and include_normals:
        mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=False, maintain_order=True)
        vertex_normals = mesh.vertex_normals
        face_normals = mesh.face_normals
    return {
        'mesh_names': np.array(names),
        'vertices': vertices,
        'faces': all_faces,
        'vertex_normals': vertex_normals,
        'face_normals': face_normals,
        'vertex_bias': np.array(vertex_bias),
        'face_bias': np.array(face_bias) if include_faces else None,
    }


class GlbParser(AbstractParser):

    @classmethod
    def load(cls, filepath: str, limits: Optional[InputLimits]=None, **kwargs) -> Asset:
        """Load a .glb/.gltf file as a plain mesh (merged mesh parts).

        Any skin, joints or animation in the file are ignored: rigs are output
        only. Pass ``limits`` for untrusted (uploaded) files; see ``glb_io.limits``.
        """
        scene = load_scene(filepath, limits=limits)
        mesh = _scene_mesh(scene)
        return Asset(
            vertices=mesh['vertices'],
            faces=mesh['faces'],
            vertex_normals=mesh['vertex_normals'],
            face_normals=mesh['face_normals'],
            vertex_bias=mesh['vertex_bias'],
            face_bias=mesh['face_bias'],
            mesh_names=mesh['mesh_names'],
            matrix_world=np.eye(4),
        )

    @classmethod
    def export(cls, asset: Asset, filepath: str, **kwargs):
        """Export .ply or .glb (skinned when the asset has joints and skin)."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.ply':
            cls.export_ply(asset, filepath, **kwargs)
        elif ext == '.glb':
            cls.export_asset(asset, filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {ext}; expected .glb or .ply")

    @classmethod
    def export_ply(
        cls,
        asset: Asset,
        filepath: str,
        use_pc: bool=False,
        render_skin_id: Optional[int]=None,
    ):
        """Export the asset as an .ply file. This will ignore skeleton and skinning."""
        import open3d as o3d
        asset._build_bias()
        if asset.vertices is None or asset.vertex_bias is None:
            raise ValueError("do not have vertices or vertex_bias")
        if not filepath.lower().endswith('.ply'):
            filepath += ".ply"
        faces = None if use_pc else asset.faces
        cls._safe_make_dir(filepath)
        if render_skin_id is not None:
            if asset.skin is None:
                raise ValueError("render_skin_id is not None, but skin of asset is None")
            colors = np.stack([
                asset.skin[:, render_skin_id],
                np.zeros(asset.N),
                1-asset.skin[:, render_skin_id],
            ], axis=1)
        else:
            colors = None
        if faces is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(asset.vertices)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(asset.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_triangle_mesh(filepath, mesh)

    @classmethod
    def export_asset(
        cls,
        asset: Asset,
        filepath: str,
        group_per_vertex: int=-1,
        do_not_normalize: bool=False,
        auto_ground: bool=False,
        **kwargs,
    ):
        """Export the asset's own mesh (no materials) skinned to its skeleton as GLB."""
        asset._build_bias()
        if asset.vertices is None or asset.vertex_bias is None:
            raise ValueError("do not have vertices or vertex_bias")
        joints, parents = asset.joints, asset.parents
        if joints is None or parents is None:
            raise ValueError("GLB export requires joints and parents")
        names = asset.mesh_names if asset.mesh_names is not None else [f"mesh_{i}" for i in range(asset.P)]
        scene = Scene(Document(
            asset_mesh_glb_bytes(_to_y_up(asset.vertices), asset.faces, asset.vertex_bias, names),
            name='asset.glb'))
        skin = asset.skin
        if skin is None:
            skin = np.zeros((asset.N, asset.J), dtype=np.float32)
        joint_names = asset.joint_names if asset.joint_names is not None else [f"unknown_{i}" for i in range(asset.J)]
        _export(scene, filepath, joints, parents, joint_names, skin,
                group_per_vertex=group_per_vertex, normalize=not do_not_normalize,
                auto_ground=auto_ground, armature_name=asset.armature_name or 'Armature')

    @classmethod
    def _safe_make_dir(cls, path: str):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _export(scene: Scene, filepath: str, joints_z_up: ndarray, parents, joint_names, skin: ndarray,
            group_per_vertex: int, normalize: bool, auto_ground: bool, armature_name: str='Armature'):
    if os.path.splitext(filepath)[1].lower() != '.glb':
        raise ValueError(f"Rigged export supports .glb only, got: {filepath}")
    offset = None
    if auto_ground:
        # Shift everything so the bounding-box bottom sits on the ground plane
        # (Blender Z=0, glTF Y=0), keeping skinning and relative transforms.
        vertices = scene.vertices()
        if len(vertices):
            min_y = float(vertices[:, 1].min())
            if abs(min_y) >= 1e-6:
                offset = np.array([0., -min_y, 0.])
    export_skinned_glb(
        scene, filepath, _to_y_up(joints_z_up), parents, [str(n) for n in joint_names], skin,
        group_per_vertex=group_per_vertex, normalize=normalize, offset=offset,
        armature_name=armature_name,
    )


def _umeyama_similarity(src: ndarray, tgt: ndarray) -> ndarray:
    assert src.shape == tgt.shape
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    # cross-covariance
    C = (src_c.T @ tgt_c) / n
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = (src_c ** 2).sum() / n
    scale = S.sum() / var_src
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def _pca_similarity(
    src: ndarray,
    tgt: ndarray,
    max_points: int=4096,
) -> ndarray:
    if src.shape[0] > max_points:
        src = src[np.random.choice(src.shape[0], max_points, replace=False)]
    if tgt.shape[0] > max_points:
        tgt = tgt[np.random.choice(tgt.shape[0], max_points, replace=False)]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    U_src, _, _ = np.linalg.svd(src_c.T @ src_c)
    U_tgt, _, _ = np.linalg.svd(tgt_c.T @ tgt_c)
    R = U_tgt @ U_src.T
    if np.linalg.det(R) < 0:
        U_tgt[:, -1] *= -1
        R = U_tgt @ U_src.T
    scale = np.sqrt((tgt_c ** 2).sum() / (src_c ** 2).sum())
    t = tgt_mean - scale * R @ src_mean
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def estimate_similarity_transform(
    src: ndarray,
    tgt: ndarray,
    max_points: int=4096,
) -> ndarray:
    """
    src: (N, 3)
    tgt: (M, 3)
    return: (4, 4) similarity transform matrix
    """
    if src.shape[0] == tgt.shape[0]:
        return _umeyama_similarity(src, tgt)
    return _pca_similarity(src, tgt, max_points)


def transfer_rigging(
    source_asset: Asset,
    target_path: str,
    export_path: str,
    group_per_vertex: int=4,
    auto_ground: bool=False,
    max_points: Optional[int]=None,
    do_not_normalize: bool=False,
    limits: Optional[InputLimits]=None,
    **kwargs,
):
    """Map a predicted (normalized-space) rig back onto the original file and export GLB.

    The source asset's mesh is aligned to the reloaded target mesh with a
    similarity transform; joints follow it and skin weights are transferred by
    nearest vertex. Only the skinned mesh is exported (no materials/textures).
    Pass ``limits`` when ``target_path`` is an untrusted upload.
    """
    assert source_asset.matrix_local is not None
    assert source_asset.parents is not None
    source_vertices = source_asset.vertices
    source_joints = source_asset.joints
    assert source_vertices is not None and source_joints is not None

    scene = load_scene(target_path, limits=limits)
    target_vertices = _to_z_up(scene.vertices())
    if len(target_vertices) == 0:
        raise ValueError(f"No mesh vertices found in {target_path}")

    T = estimate_similarity_transform(src=source_vertices, tgt=target_vertices, max_points=max_points or 4096)
    target_joints = source_joints @ T[:3, :3].T + T[:3, 3]

    J = len(source_joints)
    if source_asset.skin is not None:
        from scipy.spatial import cKDTree
        mapped = source_vertices @ T[:3, :3].T + T[:3, 3]
        _, idx = cKDTree(mapped).query(target_vertices, k=1)
        skin = source_asset.skin[idx]
    else:
        skin = np.zeros((len(target_vertices), J), dtype=np.float32)

    joint_names = source_asset.joint_names if source_asset.joint_names is not None else [f"unknown_{i}" for i in range(J)]
    _export(scene, export_path, target_joints, source_asset.parents, joint_names, skin,
            group_per_vertex=group_per_vertex, normalize=not do_not_normalize,
            auto_ground=auto_ground, armature_name=source_asset.armature_name or 'Armature')
