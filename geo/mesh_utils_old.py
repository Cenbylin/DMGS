# Copyright (c) 2019,20-21-22-23 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
# from ..batch import tile_to_packed, packed_to_padded, get_first_idx
__all__ = [
    'face_areas',
    'packed_face_areas',
    'sample_points',
    'packed_sample_points',
    'face_normals',
    'subdivide_trianglemesh',
    'vertex_tangents'
]


def _base_face_areas(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas


def _base_sample_points_selected_faces(face_vertices, face_features=None):
    """Base function to sample points over selected faces.
       The coordinates of the face vertices are interpolated to generate new samples.

    Args:
        face_vertices (tuple of torch.Tensor):
            Coordinates of vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
        face_features (tuple of torch.Tensor, Optional):
            Features of face vertices, corresponding to selected faces to sample from.
            A tuple of 3 entries corresponding to each of the face vertices.
            Each entry is a torch.Tensor of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.

    Returns:
        (torch.Tensor, torch.Tensor):
            Sampled point coordinates of shape :math:`(\\text{batch_size}, \\text{num_samples}, 3)`.
            Sampled points interpolated features of shape
            :math:`(\\text{batch_size}, \\text{num_samples}, \\text{feature_dim})`.
            If `face_vertices_features` arg is not specified, the returned interpolated features are None.
    """

    face_vertices0, face_vertices1, face_vertices2 = face_vertices

    sampling_shape = tuple(int(d) for d in face_vertices0.shape[:-1]) + (1,)
    # u is proximity to middle point between v1 and v2 against v0.
    # v is proximity to v2 against v1.
    #
    # The probability density for u should be f_U(u) = 2u.
    # However, torch.rand use a uniform (f_X(x) = x) distribution,
    # so using torch.sqrt we make a change of variable to have the desired density
    # f_Y(y) = f_X(y ^ 2) * |d(y ^ 2) / dy| = 2y
    u = torch.sqrt(torch.rand(sampling_shape,
                              device=face_vertices0.device,
                              dtype=face_vertices0.dtype))

    v = torch.rand(sampling_shape,
                   device=face_vertices0.device,
                   dtype=face_vertices0.dtype)
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * face_vertices0 + w1 * face_vertices1 + w2 * face_vertices2

    features = None
    if face_features is not None:
        face_features0, face_features1, face_features2 = face_features
        features = w0 * face_features0 + w1 * face_features1 + \
            w2 * face_features2

    return points, features


def face_areas(vertices, faces):
    """Compute the areas of each face of triangle meshes.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes,
            of shape :math:`(\\text{batch_size}, \\text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            the faces of the meshes, of shape :math:`(\\text{num_faces}, 3)`.

    Returns:
        (torch.Tensor):
            the face areas of same type as vertices and of shape
            :math:`(\\text{batch_size}, \\text{num_faces})`.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("face_areas is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))

    areas = _base_face_areas(face_v_0, face_v_1, face_v_2)

    return areas.squeeze(-1)


def sample_points(vertices, faces, num_samples, areas=None, face_features=None):
    r"""Uniformly sample points over the surface of triangle meshes.

    First face on which the point is sampled is randomly selected,
    with the probability of selection being proportional to the area of the face.
    then the coordinate on the face is uniformly sampled.

    If ``face_features`` is defined for the mesh faces,
    the sampled points will be returned with interpolated features as well,
    otherwise, no feature interpolation will occur.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes, of shape
            :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            The faces of the mesh, of shape :math:`(\text{num_faces}, 3)`.
        num_samples (int):
            The number of point sampled per mesh.
        areas (torch.Tensor, optional):
            The areas of each face, of shape :math:`(\text{batch_size}, \text{num_faces})`,
            can be preprocessed, for fast on-the-fly sampling,
            will be computed if None (default).
        face_features (torch.Tensor, optional):
            Per-vertex-per-face features, matching ``faces`` order,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`.
            For example:

                1. Texture uv coordinates would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`.
                2. RGB color values would be of shape
                   :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.

            When specified, it is used to interpolate the features for new sampled points.

    See also:
        :func:`~kaolin.ops.mesh.index_vertices_by_faces` for conversion of features defined per vertex
        and need to be converted to per-vertex-per-face shape of :math:`(\text{num_faces}, 3)`.

    Returns:
        (torch.Tensor, torch.LongTensor, (optional) torch.Tensor):
            the pointclouds of shape :math:`(\text{batch_size}, \text{num_samples}, 3)`,
            and the indexes of the faces selected,
            of shape :math:`(\text{batch_size}, \text{num_samples})`.

            If ``face_features`` arg is specified, then the interpolated features of sampled points of shape
            :math:`(\text{batch_size}, \text{num_samples}, \text{feature_dim})` are also returned.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("sample_points is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)        # (num_faces, 3) -> tuple of (num_faces,)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))  # (batch_size, num_faces, 3)
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))  # (batch_size, num_faces, 3)

    if areas is None:
        areas = _base_face_areas(face_v_0, face_v_1, face_v_2).squeeze(-1)
    areas[areas.isnan()] = 0.
    face_dist = torch.distributions.Categorical(areas)
    face_choices = face_dist.sample([num_samples]).transpose(0, 1)
    _face_choices = face_choices.unsqueeze(-1).repeat(1, 1, 3)
    v0 = torch.gather(face_v_0, 1, _face_choices)  # (batch_size, num_samples, 3)
    v1 = torch.gather(face_v_1, 1, _face_choices)  # (batch_size, num_samples, 3)
    v2 = torch.gather(face_v_2, 1, _face_choices)  # (batch_size, num_samples, 3)
    face_vertices_choices = (v0, v1, v2)

    # UV coordinates are available, make sure to calculate them for sampled points as well
    face_features_choices = None
    if face_features is not None:
        feat_dim = face_features.shape[-1]
        # (num_faces, 3) -> tuple of (num_faces,)
        _face_choices = face_choices[..., None, None].repeat(1, 1, 3, feat_dim)
        face_features_choices = torch.gather(face_features, 1, _face_choices)
        face_features_choices = tuple(
            tmp_feat.squeeze(2) for tmp_feat in torch.split(face_features_choices, 1, dim=2))

    points, point_features = _base_sample_points_selected_faces(
        face_vertices_choices, face_features_choices)

    if point_features is not None:
        return points, face_choices, point_features
    else:
        return points, face_choices


def face_normals(face_vertices, unit=False):
    r"""Calculate normals of triangle meshes. Left-hand rule convention is used for picking normal direction.

        Args:
            face_vertices (torch.Tensor):
                of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.
            unit (bool):
                if true, return normals as unit vectors. Default: False.
        Returns:
            (torch.FloatTensor):
                face normals, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`
        """
    if face_vertices.shape[-2] != 3:
        raise NotImplementedError("face_normals is only implemented for triangle meshes")
    # Note: Here instead of using the normals from vertexlist2facelist we compute it from scratch
    edges_dist0 = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    edges_dist1 = face_vertices[:, :, 2] - face_vertices[:, :, 0]
    face_normals = torch.cross(edges_dist0, edges_dist1, dim=2)

    if unit:
        face_normals_length = face_normals.norm(dim=2, keepdim=True)
        face_normals = face_normals / (face_normals_length + 1e-10)

    return face_normals


def _get_adj_verts(edges_ex2, v):
    """Get sparse adjacency matrix for vertices given edges"""
    adj_sparse_idx = torch.cat([edges_ex2, torch.flip(edges_ex2, [1])])
    adj_sparse_idx = torch.unique(adj_sparse_idx, dim=0)

    values = torch.ones(
        adj_sparse_idx.shape[0], device=edges_ex2.device).float()
    adj_sparse = torch.sparse.FloatTensor(
        adj_sparse_idx.t(), values, torch.Size([v, v]))
    return adj_sparse


def _get_alpha(n):
    """Compute weight alpha based on number of neighboring vertices following Loop Subdivision"""
    n = n.float()
    alpha = (5.0 / 8 - (3.0 / 8 + 1.0 / 4 * torch.cos(2 * math.pi / n)) ** 2) / n
    alpha[n == 3] = 3 / 16

    return alpha


def subdivide_trianglemesh(vertices, faces, iterations, alpha=None):
    r"""Subdivide triangular meshes following the scheme of Loop subdivision proposed in 
    `Smooth Subdivision Surfaces Based on Triangles`_. 
    If the smoothing factor alpha is not given, this function performs exactly as Loop subdivision.
    Elsewise the vertex position is updated using the given per-vertex alpha value, which is 
    differentiable and the alpha carries over to subsequent subdivision iterations. Higher alpha leads
    to smoother surfaces, and a vertex with alpha = 0 will not change from its initial position 
    during the subdivision. Thus, alpha can be learnable to preserve sharp geometric features in contrast to 
    the original Loop subdivision.
    For more details and example usage in learning, see `Deep Marching Tetrahedra\: a Hybrid 
    Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.

    Args:
        vertices (torch.Tensor): batched vertices of triangle meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor): unbatched triangle mesh faces, of shape
                              :math:`(\text{num_faces}, 3)`.
        iterations (int): number of subdivision iterations.
        alpha (optional, torch.Tensor): batched per-vertex smoothing factor, alpha, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.

    Returns:
        (torch.Tensor, torch.LongTensor): 
            - batched vertices of triangle meshes, of shape
                                 :math:`(\text{batch_size}, \text{new_num_vertices}, 3)`.
            - unbatched triangle mesh faces, of shape
                              :math:`(\text{num_faces} \cdot 4^\text{iterations}, 3)`.

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...                           [1, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2],[0, 1, 3],[0, 2, 3],[1, 2, 3]], dtype=torch.long)
        >>> alpha = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        >>> new_vertices, new_faces = subdivide_trianglemesh(vertices, faces, 1, alpha)
        >>> new_vertices
        tensor([[[0.0000, 0.0000, 0.0000],
                 [1.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000],
                 [0.3750, 0.1250, 0.1250],
                 [0.1250, 0.3750, 0.1250],
                 [0.1250, 0.1250, 0.3750],
                 [0.3750, 0.3750, 0.1250],
                 [0.3750, 0.1250, 0.3750],
                 [0.1250, 0.3750, 0.3750]]])
        >>> new_faces
        tensor([[1, 7, 4],
                [0, 4, 5],
                [2, 5, 7],
                [5, 4, 7],
                [1, 8, 4],
                [0, 4, 6],
                [3, 6, 8],
                [6, 4, 8],
                [2, 9, 5],
                [0, 5, 6],
                [3, 6, 9],
                [6, 5, 9],
                [2, 9, 7],
                [1, 7, 8],
                [3, 8, 9],
                [8, 7, 9]])
                
    .. _Smooth Subdivision Surfaces Based on Triangles:
            https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thesis-10.pdf    

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """
    init_alpha = alpha
    for i in range(iterations):
        device = vertices.device
        b, v, f = vertices.shape[0], vertices.shape[1], faces.shape[0]

        edges_fx3x2 = faces[:, [[0, 1], [1, 2], [2, 0]]]
        edges_fx3x2_sorted, _ = torch.sort(edges_fx3x2.reshape(edges_fx3x2.shape[0] * edges_fx3x2.shape[1], 2), -1)
        all_edges_face_idx = torch.arange(edges_fx3x2.shape[0], device=device).unsqueeze(-1).expand(-1, 3).reshape(-1)
        edges_ex2, inverse_indices, counts = torch.unique(
            edges_fx3x2_sorted, dim=0, return_counts=True, return_inverse=True)

        # To compute updated vertex positions, first compute alpha for each vertex
        # TODO(cfujitsang): unify _get_adj_verts with adjacency_matrix
        adj_sparse = _get_adj_verts(edges_ex2, v)
        n = torch.sparse.sum(adj_sparse, 0).to_dense().view(-1, 1)
        if init_alpha is None:
            alpha = (_get_alpha(n) * n).unsqueeze(0)
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1)

        adj_verts_sum = torch.bmm(adj_sparse.unsqueeze(0), vertices)
        vertices_new = (1 - alpha) * vertices + alpha / n * adj_verts_sum

        e = edges_ex2.shape[0]
        edge_points = torch.zeros((b, e, 3), device=device)  # new point for every edge
        edges_fx3 = inverse_indices.reshape(f, 3) + v
        alpha_points = torch.zeros((b, e, 1), device=device)

        mask_e = (counts == 2)

        # edge points on boundary is computed as midpoint
        if torch.sum(~mask_e) > 0:
            edge_points[:, ~mask_e] += torch.mean(vertices[:,
                                                  edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 3), 2)
            alpha_points[:, ~mask_e] += torch.mean(alpha[:, edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 1), 2)

        counts_f = counts[inverse_indices]
        mask_f = (counts_f == 2)
        group = inverse_indices[mask_f]
        _, indices = torch.sort(group)
        edges_grouped = all_edges_face_idx[mask_f][indices]
        edges_face_idx = torch.stack([edges_grouped[::2], edges_grouped[1::2]], dim=-1)
        e_ = edges_face_idx.shape[0]
        edges_face = faces[edges_face_idx.reshape(-1), :].reshape(-1, 2, 3)
        edges_vert = vertices[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 3)
        edges_vert = torch.cat([edges_vert, vertices[:, edges_ex2[mask_e].reshape(-1),
                               :].reshape(b, -1, 2, 3)], 2).mean(2)

        alpha_vert = alpha[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 1)
        alpha_vert = torch.cat([alpha_vert, alpha[:, edges_ex2[mask_e].reshape(-1),
                               :].reshape(b, -1, 2, 1)], 2).mean(2)

        edge_points[:, mask_e] += edges_vert
        alpha_points[:, mask_e] += alpha_vert

        alpha = torch.cat([alpha, alpha_points], 1)
        vertices = torch.cat([vertices_new, edge_points], 1)
        faces = torch.cat([faces, edges_fx3], 1)
        faces = faces[:, [[1, 4, 3], [0, 3, 5], [2, 5, 4], [5, 3, 4]]].reshape(-1, 3)
    return vertices, faces

def vertex_tangents(faces, face_vertices, face_uvs, vertex_normals):
    r"""Compute vertex tangents.

    The vertex tangents are useful to apply normal maps during rendering.

    .. seealso::

        https://en.wikipedia.org/wiki/Normal_mapping#Calculating_tangent_space

    Args:
       faces (torch.LongTensor): unbatched triangle mesh faces, of shape
                                 :math:`(\text{num_faces}, 3)`.
       face_vertices (torch.Tensor): unbatched triangle face vertices, of shape
                                     :math:`(\text{num_faces}, 3, 3)`.
       face_uvs (torch.Tensor): unbatched triangle UVs, of shape
                                :math:`(\text{num_faces}, 3, 2)`.
       vertex_normals (torch.Tensor): unbatched vertex normals, of shape
                                      :math:`(\text{num_vertices}, 3)`.

    Returns:
       (torch.Tensor): The vertex tangents, of shape :math:`(\text{num_vertices, 3})`
    """
    # This function is strongly inspired by
    # https://github.com/NVlabs/nvdiffrec/blob/main/render/mesh.py#L203
    tangents = torch.zeros_like(vertex_normals)

    face_uvs0, face_uvs1, face_uvs2 = torch.split(face_uvs, 1, dim=-2)
    fv0, fv1, fv2 = torch.split(face_vertices, 1, dim=-2)
    uve1 = face_uvs1 - face_uvs0
    uve2 = face_uvs2 - face_uvs0
    pe1 = (fv1 - fv0).squeeze(-2)
    pe2 = (fv2 - fv0).squeeze(-2)

    nom = pe1 * uve2[..., 1] - pe2 * uve1[..., 1]
    denom = uve1[..., 0] * uve2[..., 1] - uve1[..., 1] * uve2[..., 0]
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(
        denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
    )
    vn_idx = torch.split(faces, 1, dim=-1)
    indexing_dim = 0 if face_vertices.ndim == 3 else 1
    # TODO(cfujitsang): optimizable?
    for i in range(3):
        idx = vn_idx[i].repeat(1, 3)
        tangents.scatter_add_(indexing_dim, idx, tang)
    # Normalize and make sure tangent is perpendicular to normal
    tangents = torch.nn.functional.normalize(tangents, dim=1)
    tangents = torch.nn.functional.normalize(
        tangents -
        torch.sum(tangents * vertex_normals, dim=-1, keepdim=True) *
        vertex_normals
    )

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents
