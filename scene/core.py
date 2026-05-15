import torch
import geo.mesh_utils as mesh_utils
from pytorch3d.transforms import matrix_to_quaternion

try:  # only for ablation study
    from torch_scatter import scatter_add
    from torch_geometric.nn import knn
except Exception as e:
    pass

def bind_gs_to_face(verts, faces, bc_coords, rad_base, spatial_scale,
                    mlp_texture, adaptive_cov=True):
    device = 'cuda'
    # triangle space definition & data shape
    face_vert = verts[faces]
    v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
    v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
    face_x = torch.div(v1v2, v1v2_len)
    face_normals = mesh_utils.face_normals(verts, faces, unit=True)
    face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
    rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
    n_face = faces.shape[0]
    N_gs_per_face = bc_coords.shape[0]
    N_gs = n_face * N_gs_per_face
    
    # gaussian means
    gs_xyz = torch.matmul(
        bc_coords.view(1, -1, 3), face_vert)  # [1, n_inner_gs, 3] @ [n_triangle, 3, 3]
    gs_xyz = gs_xyz.view(-1, 3)  # [n_triangle*n_inner_gs, 3]
    
    # gaussian covariance (triangle space)
    with torch.no_grad():
        v1v2_xlen = v1v2_len.view(n_face)
        v1v3 = face_vert[:, 2, :] - face_vert[:, 0, :]
        if adaptive_cov:
            # affine matrix M: from equilateral to current triangle, ME = A 
            # 1. no shift, no scale/direction change at x-axis, so
            # M must be [[1, m1], [0, m2]], with m1, m2 to be solved.
            # 2. assume only the third pair of points E_x/y, A_x/y change,
            # m1 = (A_x-E_x)/E_y, m2 = A_y/E_y
            A = torch.stack([(v1v3*face_x).sum(-1), (v1v3*face_y).sum(-1)], dim=-1)
            E = torch.stack([v1v2_xlen/2., v1v2_xlen * ((3**0.5)/2)], dim=-1)
            M_2d = torch.zeros((n_face, 2, 2), device=device)
            M_2d[:, 0, 0] = 1.
            M_2d[:, 0, 1] = torch.div((A[:, 0] - E[:, 0]), E[:, 1])
            M_2d[:, 1, 1] = torch.div(A[:, 1], E[:, 1])
        else:  # ablation study
            M_2d = torch.eye(2, device=device).view(1, 2, 2)

        # L=M R_init S, **2D triangle space**, here R_init=I
        s_scalar = (v1v2_xlen * rad_base)  # s_scalar is std-deviation
        L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_face, 1, 1]
        cov_matrix = torch.bmm(L, L.transpose(1, 2))
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # std-deviation; ==self.scaling_activation(self._scaling), i.e. already activated
        scale2D = torch.sqrt(eigenvalues.clamp_min(1e-30))
        _scaling_local = torch.cat([scale2D, 
            torch.full_like(scale2D[:, :1], spatial_scale * 1e-6)], dim=-1)

        _rotation_local = torch.zeros(n_face, 3, 3, device='cuda')
        _rotation_local[:, :2, :2] = eigenvectors
        _rotation_local[:,  2,  2] = torch.det(eigenvectors)  # ensure a legal rotation matrix

    # gaussian rotation & scale, shared by face gaussians
    _scaling = _scaling_local  #  * self.scale_factor
    _rotation = matrix_to_quaternion(torch.bmm(rot_t2w, _rotation_local))
    _scaling_acted = _scaling.view(n_face, 1, 3).expand(n_face, N_gs_per_face, 3).reshape(N_gs, 3)
    _rotation = _rotation.view(n_face, 1, 4).expand(n_face, N_gs_per_face, 4).reshape(N_gs, 4)

    # gaussian colors & opacities
    # features = torch.zeros(N_gs, (self.max_sh_degree+1) ** 2, 3, device='cuda')  # debug
    # features[0::3, :1, 0], features[1::3, :1, 1], features[2::3, :1, 2] = 1., 1., 1.
    features = mlp_texture.sample_noact(gs_xyz)  # view(N_gs, SH_dim, 3)
    opacities_acted = torch.full((N_gs, 1), 0.999999, device=device)

    return gs_xyz, _scaling_acted, _rotation, features, opacities_acted

def bind_2dgs_to_face(verts, faces, bc_coords, rad_base, spatial_scale,
                      mlp_texture, adaptive_cov=True):
    device = 'cuda'
    # triangle space definition & data shape
    face_vert = verts[faces]
    v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
    v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
    face_x = torch.div(v1v2, v1v2_len)
    face_normals = mesh_utils.face_normals(verts, faces, unit=True)
    face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
    rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
    n_face = faces.shape[0]
    N_gs_per_face = bc_coords.shape[0]
    N_gs = n_face * N_gs_per_face
    
    # gaussian means
    gs_xyz = torch.matmul(
        bc_coords.view(1, -1, 3), face_vert)  # [1, n_inner_gs, 3] @ [n_triangle, 3, 3]
    gs_xyz = gs_xyz.view(-1, 3)  # [n_triangle*n_inner_gs, 3]
    
    # gaussian covariance (triangle space)
    with torch.no_grad():
        v1v2_xlen = v1v2_len.view(n_face)
        v1v3 = face_vert[:, 2, :] - face_vert[:, 0, :]
        if adaptive_cov:
            # affine matrix M: from equilateral to current triangle, ME = A 
            # 1. no shift, no scale/direction change at x-axis, so
            # M must be [[1, m1], [0, m2]], with m1, m2 to be solved.
            # 2. assume only the third pair of points E_x/y, A_x/y change,
            # m1 = (A_x-E_x)/E_y, m2 = A_y/E_y
            A = torch.stack([(v1v3*face_x).sum(-1), (v1v3*face_y).sum(-1)], dim=-1)
            E = torch.stack([v1v2_xlen/2., v1v2_xlen * ((3**0.5)/2)], dim=-1)
            M_2d = torch.zeros((n_face, 2, 2), device=device)
            M_2d[:, 0, 0] = 1.
            M_2d[:, 0, 1] = torch.div((A[:, 0] - E[:, 0]), E[:, 1])
            M_2d[:, 1, 1] = torch.div(A[:, 1], E[:, 1])
        else:  # ablation study
            M_2d = torch.eye(2, device=device).view(1, 2, 2)

        # L=M R_init S, **2D triangle space**, here R_init=I
        s_scalar = (v1v2_xlen * rad_base)  # s_scalar is std-deviation
        L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_face, 1, 1]
        cov_matrix = torch.bmm(L, L.transpose(1, 2))
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # std-deviation; ==self.scaling_activation(self._scaling), i.e. already activated
        scale2D = torch.sqrt(eigenvalues.clamp_min(1e-30))

        # R_t2w = [[a, -b], [b, a]], first base is (a, b)
        rot2D = torch.nn.functional.normalize(eigenvectors[:, :, 0], dim=1)  # (a, b), eigenvectors[:,:,1] may be a unwanted [b,-a] 

    # gaussian rotation & scale, shared by face gaussians
    _scaling_acted = scale2D.view(n_face, 1, 2).expand(n_face, N_gs_per_face, 2).reshape(N_gs, 2)
    _rotation = rot2D.view(n_face, 1, 2).expand(n_face, N_gs_per_face, 2).reshape(N_gs, 2)

    # gaussian colors & opacities
    features = mlp_texture.sample_noact(gs_xyz)  # view(N_gs, SH_dim, 3)
    opacities_acted = torch.full((N_gs, 1), 0.999999, device=device)

    return gs_xyz, _scaling_acted, _rotation, features, opacities_acted

#######################
# for ablation study
#######################
def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, k: int = 3):
    """ refer to torch_geometric.nn.unpool.knn_interpolate
        INPUT   x: features attached to reference nodes to be interpolated
                pos_x: location of reference nodes, shoule be the particle@t
                pos_y: location of the query points
                K: #. nearest neighbours
    """
    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, batch_x=None, batch_y=None,
                            num_workers=1)
        y_idx, x_idx = assign_index[0], assign_index[1]
    diff = pos_x[x_idx] - pos_y[y_idx]
    squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
    weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
    y = scatter_add(x.flatten(start_dim=1)[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))
    return y


def bind_gs_to_face_knn(verts, faces, bc_coords, rad_base, spatial_scale,
                        xyz_prev, feat_prev, adaptive_cov=True):
    device = 'cuda'
    # triangle space definition & data shape
    face_vert = verts[faces]
    v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
    v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
    face_x = torch.div(v1v2, v1v2_len)
    face_normals = mesh_utils.face_normals(verts, faces, unit=True)
    face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
    rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
    n_face = faces.shape[0]
    N_gs_per_face = bc_coords.shape[0]
    N_gs = n_face * N_gs_per_face
    
    # gaussian means
    gs_xyz = torch.matmul(
        bc_coords.view(1, -1, 3), face_vert)  # [1, n_inner_gs, 3] @ [n_triangle, 3, 3]
    gs_xyz = gs_xyz.view(-1, 3)  # [n_triangle*n_inner_gs, 3]
    
    # gaussian covariance (triangle space)
    with torch.no_grad():
        v1v2_xlen = v1v2_len.view(n_face)
        v1v3 = face_vert[:, 2, :] - face_vert[:, 0, :]
        if adaptive_cov:
            # affine matrix M: from equilateral to current triangle, ME = A 
            # 1. no shift, no scale/direction change at x-axis, so
            # M must be [[1, m1], [0, m2]], with m1, m2 to be solved.
            # 2. assume only the third pair of points E_x/y, A_x/y change,
            # m1 = (A_x-E_x)/E_y, m2 = A_y/E_y
            A = torch.stack([(v1v3*face_x).sum(-1), (v1v3*face_y).sum(-1)], dim=-1)
            E = torch.stack([v1v2_xlen/2., v1v2_xlen * ((3**0.5)/2)], dim=-1)
            M_2d = torch.zeros((n_face, 2, 2), device=device)
            M_2d[:, 0, 0] = 1.
            M_2d[:, 0, 1] = torch.div((A[:, 0] - E[:, 0]), E[:, 1])
            M_2d[:, 1, 1] = torch.div(A[:, 1], E[:, 1])
        else:  # ablation study
            M_2d = torch.eye(2, device=device).view(1, 2, 2)

        # L=M R_init S, **2D triangle space**, here R_init=I
        s_scalar = (v1v2_xlen * rad_base)  # s_scalar is std-deviation
        L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_face, 1, 1]
        cov_matrix = torch.bmm(L, L.transpose(1, 2))
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # std-deviation; ==self.scaling_activation(self._scaling), i.e. already activated
        scale2D = torch.sqrt(eigenvalues.clamp_min(1e-30))
        _scaling_local = torch.cat([scale2D, 
            torch.full_like(scale2D[:, :1], spatial_scale * 1e-6)], dim=-1)

        _rotation_local = torch.zeros(n_face, 3, 3, device='cuda')
        _rotation_local[:, :2, :2] = eigenvectors
        _rotation_local[:,  2,  2] = torch.det(eigenvectors)  # ensure a legal rotation matrix

    # gaussian rotation & scale, shared by face gaussians
    _scaling = _scaling_local  #  * self.scale_factor
    _rotation = matrix_to_quaternion(torch.bmm(rot_t2w, _rotation_local))
    _scaling_acted = _scaling.view(n_face, 1, 3).expand(n_face, N_gs_per_face, 3).reshape(N_gs, 3)
    _rotation = _rotation.view(n_face, 1, 4).expand(n_face, N_gs_per_face, 4).reshape(N_gs, 4)

    # gaussian colors & opacities
    # features = torch.zeros(N_gs, (self.max_sh_degree+1) ** 2, 3, device='cuda')  # debug
    # features[0::3, :1, 0], features[1::3, :1, 1], features[2::3, :1, 2] = 1., 1., 1.
    # features = mlp_texture.sample_noact(gs_xyz)  # view(N_gs, SH_dim, 3)
    if feat_prev is None:
        features = torch.zeros(N_gs, (3+1) ** 2, 3, device='cuda') 
    else:
        with torch.no_grad():
            _, sh_dim, color_dim = feat_prev.shape
            features = knn_interpolate(feat_prev.flatten(start_dim=1), 
                                       xyz_prev, gs_xyz, k=6).view(-1, sh_dim, color_dim)
    
    opacities_acted = torch.full((N_gs, 1), 0.999999, device=device)

    return gs_xyz, _scaling_acted, _rotation, features, opacities_acted