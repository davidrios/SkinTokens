
import os
from concurrent.futures import ThreadPoolExecutor
from numpy import ndarray
from typing import Callable, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse

def assert_ndarray(arr, name: str="arr", shape: Optional[Tuple[int, ...]]=None, dtype=None):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray or None, got {type(arr)}")
    if shape is not None:
        # shape may contain None as wildcard
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}")
        for i, (exp, actual) in enumerate(zip(shape, arr.shape)):
            if exp > 0 and exp != actual:
                raise ValueError(f"{name} shape mismatch at axis {i}: expected {exp}, got {actual}")
    if dtype is not None:
        if not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")

def assert_list(arr, name: str="arr", dtype=None):
    if not isinstance(arr, list):
        raise ValueError(f"found type {type(arr)}, expect a list")
    if dtype is not None:
        for x in arr:
            if not isinstance(x, dtype):
                raise ValueError(f"found type {type(x)} in {name}, expect all to be {dtype}")

def linear_blend_skinning(
    vertices: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=1,
    value: float=1.0,
) -> ndarray:
    """
    Args:
        vertices: (N, 4-pad)
        matrix_local: (J, 4, 4)
        matrix: (J, 4, 4)
        skin: (N, J)
        pad: 0 or 1
        value: value to pad
    Returns:
        (N, 3) vertices using LBS algorithm: Skinning with dual quaternions, Kavan, 2007
    """
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    assert_ndarray(vertices, name='vertices', shape=(N, 3))
    assert_ndarray(matrix_local, name="matrix_local", shape=(J, 4, 4))
    assert_ndarray(matrix, name="matrix", shape=(J, 4, 4))
    assert_ndarray(skin, name="skin", shape=(N, J))
    assert vertices.shape[-1] + pad == 4
    # (4, N)
    padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    # (J, 4, 4)
    trans = matrix @ np.linalg.inv(matrix_local)
    weighted_per_bone_matrix = []
    # (J, N)
    mask = (skin > 0).T
    for i in range(J):
        offset = np.zeros((4, N), dtype=np.float32)
        offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
        weighted_per_bone_matrix.append(offset)
    weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
    g = np.sum(weighted_per_bone_matrix, axis=0)
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T

def cpu_workers(default_cap: int = 8) -> int:
    """Threads for parallel CPU work: ``SKINTOKENS_CPU_WORKERS``, else the
    core count up to ``default_cap`` (past that, memory bandwidth limits)."""
    raw = os.environ.get("SKINTOKENS_CPU_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, min(default_cap, os.cpu_count() or 1))


_pool: Optional[ThreadPoolExecutor] = None
_pool_size = 0


def parallel_rows(fn: Callable[[slice], None], n: int, min_chunk: int = 65536) -> None:
    """Call ``fn(rows)`` over contiguous slices covering ``range(n)``, in threads.

    For numpy work that releases the GIL and writes each slice's results into
    preallocated arrays: the slices are disjoint, so the output does not depend
    on how the rows are split. Small inputs run inline.
    """
    global _pool, _pool_size
    workers = min(cpu_workers(), max(1, n // min_chunk))
    if workers <= 1:
        fn(slice(0, n))
        return
    if _pool is None or _pool_size < workers:
        _pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rows")
        _pool_size = workers
    bounds = np.linspace(0, n, workers + 1).astype(int)
    futures = [_pool.submit(fn, slice(lo, hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]
    for fut in futures:
        fut.result()


_TOL_ZERO = np.finfo(np.float64).resolution * 100  # trimesh.util.TOL_ZERO
_TOL_MERGE = 1e-8                                   # trimesh.constants.tol.merge


def _unitize_rows(v: ndarray) -> Tuple[ndarray, ndarray]:
    """Rows scaled to unit length; rows with (near-)zero norm are left as is."""
    norm = np.sqrt(np.einsum('ij,ij->i', v, v))
    valid = norm > _TOL_ZERO
    norm[valid] = 1.0 / norm[valid]
    norm[~valid] = 1.0
    return v * norm[:, None], valid


def mesh_normals(vertices: ndarray, faces: ndarray) -> Tuple[ndarray, ndarray]:
    """``(vertex_normals, face_normals)`` (float64) as trimesh computes them.

    Face normals are the unit cross products (zero for degenerate faces);
    vertex normals are the unit sum of the adjacent face normals weighted by
    the face's corner angle at the vertex (zero for unreferenced vertices).
    Matches ``trimesh.Trimesh(..., process=False).vertex_normals/face_normals``
    up to float rounding, without trimesh's per-call cache hashing.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    face_normals = np.empty((len(f), 3))
    angles = np.empty((len(f), 3))

    def per_face(rows: slice):
        ff = f[rows]
        a, b, c = v[ff[:, 0]], v[ff[:, 1]], v[ff[:, 2]]
        ab, ac, bc = b - a, c - a, c - b
        fn, valid = _unitize_rows(np.cross(ab, ac))
        fn[~valid] = 0.0
        face_normals[rows] = fn
        # corner angles, zeroed for degenerate triangles (trimesh.triangles.angles)
        u, _ = _unitize_rows(ab)
        w, _ = _unitize_rows(ac)
        x, _ = _unitize_rows(bc)
        ang = angles[rows]
        ang[:, 0] = np.arccos(np.clip(np.einsum('ij,ij->i', u, w), -1, 1))
        ang[:, 1] = np.arccos(np.clip(-np.einsum('ij,ij->i', u, x), -1, 1))
        ang[:, 2] = np.pi - ang[:, 0] - ang[:, 1]
        ang[(ang < _TOL_MERGE).any(axis=1)] = 0.0

    # Rows are independent, so chunks give bit-identical results; numpy
    # releases the GIL in these ops, so the chunks run in parallel.
    parallel_rows(per_face, len(f))
    # Sum of corner angle * face normal per vertex, as the transpose of a
    # (F, N) matrix with one row per face. Faces with invalid normals (zeroed
    # above) contribute nothing, as trimesh drops them.
    incidence = scipy.sparse.csr_matrix(
        (angles.reshape(-1), f.reshape(-1), np.arange(0, 3 * len(f) + 1, 3)),
        shape=(len(f), len(v)))
    vertex_normals, _ = _unitize_rows(np.asarray(incidence.T @ face_normals))
    return vertex_normals, face_normals


def normals_at(vertices: ndarray, faces: ndarray, vertex_ids: ndarray, face_ids: ndarray) -> Tuple[ndarray, ndarray]:
    """``mesh_normals(vertices, faces)`` restricted to ``vertex_ids`` / ``face_ids``.

    Only the faces around the requested vertices are evaluated, so this costs a
    pass over the face indices instead of the full normal computation.
    """
    vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    f = np.asarray(faces)
    if len(vertex_ids):
        wanted = np.zeros(len(vertices), dtype=bool)
        wanted[vertex_ids] = True
        around = f[wanted[f].any(axis=1)]
        vertex_normals = mesh_normals(vertices, around)[0][vertex_ids]
    else:
        vertex_normals = np.zeros((0, 3))
    v = np.asarray(vertices, dtype=np.float64)
    picked = f[face_ids]
    a = v[picked[:, 0]]
    face_normals, valid = _unitize_rows(np.cross(v[picked[:, 1]] - a, v[picked[:, 2]] - a))
    face_normals[~valid] = 0.0
    return vertex_normals, face_normals


def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """
    Turn axis angle representation to matrix representation.
    """
    res = np.pad(scipy.spatial.transform.Rotation.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray]=None,
) -> Tuple[ndarray, ndarray, ndarray]:
    '''
    Randomly pick samples proportional to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    
    Args:
        mask: (num_faces,), only sample points on the faces where value is True.
    Return:
        vertex_samples: sampled vertices
        
        original_face_index: on which face is sampled
        
        random_lengths: sampled vectors on face
    '''
    original_face_indices = np.arange(len(faces))
    # sample according to mask
    if mask is not None:
        assert_ndarray(arr=mask, name="mask", shape=(faces.shape[0],))
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]

    # get face area (row-wise, so computed in parallel chunks bit-identically)
    face_weight = np.empty(len(faces), dtype=np.result_type(vertices.dtype, np.float32))

    def areas(rows: slice):
        f = faces[rows]
        offset_0 = vertices[f[:, 1]] - vertices[f[:, 0]]
        offset_1 = vertices[f[:, 2]] - vertices[f[:, 0]]
        # TODO: change to correct uniform sampling...
        face_weight[rows] = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)

    parallel_rows(areas, len(faces))
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # face_weight = np.cross(offset_0, offset_1, axis=-1)
    # face_weight = (face_weight * face_weight).sum(axis=1)
    
    # weight_cum = np.cumsum(face_weight, axis=0)
    # face_pick = np.random.rand(num_samples) * weight_cum[-1]
    # face_index = np.searchsorted(weight_cum, face_pick)
    
    # map face_index back to original indices
    original_face_index = original_face_indices[face_index]
    
    # Pull the picked triangles (not every face of the mesh: there are far fewer
    # samples than faces) into the form of an origin + 2 vectors.
    picked = faces[face_index]
    tri_origins = vertices[picked[:, 0]]
    tri_vectors = vertices[picked[:, 1:]]
    tri_vectors -= tri_origins[:, None, :]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, random_lengths

def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]]
    v_vectors -= v_origins[:, np.newaxis, :]
    
    sample_vector = (v_vectors * random_lengths).sum(axis=1)
    v_samples = sample_vector + v_origins
    return v_samples

def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int]=None,
    vertex_normals: Optional[ndarray]=None,
    face_normals: Optional[ndarray]=None,
    vertex_groups: Optional[ndarray]=None,
    face_mask: Optional[ndarray]=None,
    shuffle: bool=True,
    same: bool=False,
    normals_on_demand: bool=False,
) -> Tuple[ndarray, ndarray|None, ndarray|None]:
    """
    Choose num_samples samples on the mesh and get their positions and normals.
    If vertex_group is provided, get its weights using barycentric sampling.
    
    Return:
        sampled_vertices, sampled_normals, sampled_vertex_groups
    
    Args:
        vertices: (N, 3)
        
        faces: (F, 3)
        
        num_samples: how many samples
        
        num_vertex_samples:
            At most num_vertex_samples unique vertices to be included,
            these points will be concatenated in the last (if shuffle is False).
        
        vertex_normals: (N, 3), sampled_normals will be None if not provided
        
        face_normals: (N, 3), sampled_normals will be None if not provided
        
        vertex_groups: (N, m), sampled_vertex_groups will be None if not provided
        
        face_mask:
            (F,) or (F, m), if shape is (F,), use the same mask across all
            vertex groups. Only sample on faces where value is True.
        
        shuffle: shuffle samples in the end
        
        same:
            Sample on the same locations, only useful when using mutiple
            vertex groups and mask is None or shape of (F,).

        normals_on_demand:
            Without vertex_normals/face_normals, compute the normals of just
            the sampled vertices and faces (see ``normals_at``) instead of
            returning no sampled normals.
    """

    def sampled_normals_of(perm, face_index):
        if vertex_normals is not None and face_normals is not None:
            return np.concatenate([vertex_normals[perm], face_normals[face_index]], axis=0)
        if normals_on_demand:
            return np.concatenate(normals_at(vertices, faces, perm, face_index), axis=0)
        return None

    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples cannot be larger than num_samples, found: {num_vertex_samples} > {num_samples}")
    
    def get_mask_perm(mask: Optional[ndarray]):
        if mask is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask)
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:num_vertex_samples]]
    
    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0],))
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0], -1))
            vertex_groups = vertex_groups
    
    if vertex_groups is not None:
        if face_mask is not None:
            if face_mask.ndim == 1:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            else:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0], vertex_groups.shape[1]))
        list_sampled_vertices = []
        list_sampled_normals = []
        list_sampled_vertex_groups = []
        perm = None
        _mask = None
        same = same and (face_mask is None or (face_mask is not None and face_mask.ndim != 2))
        for i in range(vertex_groups.shape[1]):
            if face_mask is not None:
                if face_mask.ndim == 1:
                    perm = get_mask_perm(faces[face_mask])
                    _mask = face_mask
                else:
                    perm = get_mask_perm(faces[face_mask[:, i]])
                    _mask = face_mask[:, i]
            else:
                perm = get_mask_perm(None)
                _mask = None
            _num_samples = num_samples - len(perm)
            
            face_vertices, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
            )
            
            list_sampled_vertices.append(np.concatenate([vertices[perm], face_vertices], axis=0))
            normals = sampled_normals_of(perm, face_index)
            if normals is not None:
                list_sampled_normals.append(normals)
            
            if same:
                g = sample_barycentric(
                    vertex_group=vertex_groups,
                    faces=faces,
                    face_index=face_index,
                    random_lengths=random_lengths,
                )
                list_sampled_vertex_groups.append(np.concatenate([vertex_groups[perm], g], axis=0))
                break
            g = sample_barycentric(
                vertex_group=vertex_groups[:, i:i+1],
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )[:, 0]
            list_sampled_vertex_groups.append(np.concatenate([vertex_groups[:, i][perm], g], axis=0))
        sampled_vertices = np.stack(list_sampled_vertices, axis=1)
        if len(list_sampled_normals) > 0:
            sampled_normals = np.stack(list_sampled_normals, axis=1)
        else:
            sampled_normals = None
        if same:
            sampled_vertex_groups = list_sampled_vertex_groups[0]
        else:
            sampled_vertex_groups = np.stack(list_sampled_vertex_groups, axis=1)
    else: # otherwise only sample vertices and normals
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            perm = get_mask_perm(faces[face_mask])
        else:
            perm = get_mask_perm(None)
        num_samples -= len(perm)
        n_vertex = vertices[perm]
        face_vertices, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            mask=face_mask,
        )
        sampled_vertices = np.concatenate([n_vertex, face_vertices], axis=0)
        sampled_normals = sampled_normals_of(perm, face_index)
        sampled_vertex_groups = None
    
    return sampled_vertices, sampled_normals, sampled_vertex_groups