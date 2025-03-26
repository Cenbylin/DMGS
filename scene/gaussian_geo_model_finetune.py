#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes

import trimesh
import open3d as o3d
import geo.mesh_utils as mesh_utils
from geo.texture import MLPTexture3D
from pytorch3d.transforms import matrix_to_quaternion

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

@torch.no_grad()
@torch.jit.script
def in_frustum(proj_matrix: torch.Tensor, verts: torch.Tensor):
    # following in_frustum() in cuda_rasterize
    # method-1 use projection matrix.
    proj_matrix_3x3 = proj_matrix[:3, :]
    # trick: we fuse the `add cube_len/2. to verts` into the proj_matrix, to save time&memory
    # so, below = (vert + cube_len/2.) @ proj_matrix_3x3 + proj_matrix_1x3
    proj_matrix_1x3 = proj_matrix[3:, :]
    p_proj = torch.matmul(verts, proj_matrix_3x3)
    p_proj += proj_matrix_1x3  #  # [N, 4] @ [4, 4], use right-mul
    w = (p_proj[:, 3:] + 1e-6)
    p_proj = p_proj[:, :3] / w
    mask = (p_proj.abs()<1.05).all(dim=-1) & (w.squeeze()>0)  # in front of camera

    return mask

class GaussianGeoModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, use_frustum):
        self.active_sh_degree = sh_degree  # important
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # added
        self.rot_t2w = torch.empty(0)
        self.use_frustum = use_frustum
        self.eval = False

    def capture(self):
        return (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            # added
            self.verts, 
            self.faces,
            self.fg_bg_nfaces,
            self.bc_coords,
            self.rad_base,
            self.thin_z_scale,
        )
    
    @torch.no_grad()
    def load(self, model_args):
        (_xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, 
         verts, faces, fg_bg_nfaces, self.bc_coords, self.rad_base, self.thin_z_scale) = model_args
        self.verts, self.faces = verts, faces
        self.eval = False
        # 1. xyz, scale, feature, opacity
        self._xyz = _xyz
        self._scaling = _scaling
        self._features_dc = _features_dc
        self._features_rest = _features_rest
        self._opacity = _opacity
        self._rotation = _rotation

        N_gs_per_face = self.bc_coords.shape[0]
        n_face = len(self.faces)
        N_gs = n_face * N_gs_per_face
        self.N_gs_per_face = N_gs_per_face
        self.fg_bg_nfaces = fg_bg_nfaces


    @torch.no_grad()
    def load_for_eval(self, model_args):
        self.eval = True  # important
        self.gs_mask = None  # no need

        (_xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, 
         verts, faces, fg_bg_nfaces, self.bc_coords, self.rad_base, self.thin_z_scale) = model_args
        self.verts, self.faces = verts, faces

        N_gs_per_face = self.bc_coords.shape[0]
        n_face = len(self.faces)
        N_gs = n_face * N_gs_per_face
        self.N_gs_per_face = N_gs_per_face
        self.fg_bg_nfaces = fg_bg_nfaces

        # 1. xyz, scale, feature, opacity
        self._xyz = _xyz
        self._scaling = _scaling
        self._features_dc = _features_dc
        self._features_rest = _features_rest
        self._opacity = _opacity
        self._rotation = _rotation
        
        # 2. rotation (quaternion and activated)
        face_normals = mesh_utils.face_normals(verts, faces, unit=True)
        face_x = torch.nn.functional.normalize(verts[faces[:, 1]] - verts[faces[:, 0]])
        face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
        self.rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
        R = self.get_rot_matrix()
        self._rotation = self.rotation_activation(matrix_to_quaternion(R))


    @torch.no_grad()
    def bind_gs_to_face(self, verts, faces, bc_coords, rad_base, 
                        mlp_texture, max_sh_degree):
        """copied from stage-2 renew_gaussian()"""
        device = 'cuda'
        # triangle space definition & data shape
        face_vert = verts[faces]
        v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
        v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
        face_x = torch.div(v1v2, v1v2_len)
        face_normals = mesh_utils.face_normals(verts, faces, unit=True)
        face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
        # self.rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
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

            A = torch.stack([(v1v3*face_x).sum(-1), (v1v3*face_y).sum(-1)], dim=-1)
            E = torch.stack([v1v2_xlen/2., v1v2_xlen/2. * (3**0.5)], dim=-1)
            M_2d = torch.zeros((n_face, 2, 2), device=device)
            M_2d[:, 0, 0] = 1.
            M_2d[:, 0, 1] = torch.div((A[:, 0] - E[:, 0]),  E[:, 1])
            M_2d[:, 1, 1] = torch.div(A[:, 1], E[:, 1])

            # L=M R_init S, **2D triangle space**, here R_init=I
            s_scalar = (v1v2_xlen * rad_base)  # s_scalar is std-deviation
            L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_tri, 1, 1]
            cov3D_L = torch.zeros((n_face, 3, 3), device=device)
            cov3D_L[:, :2, :2] = L.view(n_face, 2, 2)
            cov3D_L[:, 2, 2] = self.spatial_lr_scale * 1e-6  # flat
        
        # texture
        features = mlp_texture.sample_noact(gs_xyz).view(N_gs, 3, (max_sh_degree + 1) ** 2)
        
        _opacity = torch.full((N_gs, 1), 0.9999, device=device)
        return cov3D_L, features, _opacity


    @torch.no_grad()
    def create_from_geo_and_gs_v2(self, training_args, model_args, model_path):
        # 1. mesh & gaussians info
        (xyz, features, opacity, cov3D_L, verts, faces, fg_bg_nfaces, 
         opt_dict, self.active_sh_degree, self.spatial_lr_scale, bc_coords, 
         rad_base, scale_factor_fgbg, AABB, mlp_texture_state) = model_args

        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                                   faces=faces.detach().cpu().numpy())
        with open(f'{model_path}/mesh_init.obj', 'w') as file:
                mesh.export(file_obj=file, file_type='obj')

        # -------------------------------------------------- #
        # pre-process the mesh, bake texture to new Gaussians
        # -------------------------------------------------- #
        # options
        need_subdivision = (training_args.simplify_nface>len(faces))
        need_simplify_mesh = training_args.simplify_nface>0  # if have target nface
        need_new_bind_rule = training_args.s3_gs_per_face != bc_coords.shape[0]
        
        if need_subdivision:
            fg_nf, bg_nf = fg_bg_nfaces.tolist()

            o3d_mesh_fg = o3d.geometry.TriangleMesh()
            o3d_mesh_fg.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
            o3d_mesh_fg.triangles = o3d.utility.Vector3iVector(faces[:fg_nf].detach().cpu().numpy())
            o3d_mesh_fg = o3d_mesh_fg.remove_unreferenced_vertices()
            o3d_mesh_fg = o3d_mesh_fg.remove_degenerate_triangles()
            o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_triangles()
            o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_vertices()
            o3d_mesh_fg = o3d_mesh_fg.remove_non_manifold_edges()
            o3d_mesh_fg = o3d_mesh_fg.subdivide_loop(1)

            o3d_mesh_bg = o3d.geometry.TriangleMesh()
            o3d_mesh_bg.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
            o3d_mesh_bg.triangles = o3d.utility.Vector3iVector(faces[fg_nf:].detach().cpu().numpy())
            o3d_mesh_bg = o3d_mesh_bg.remove_unreferenced_vertices()
            o3d_mesh_bg = o3d_mesh_bg.remove_degenerate_triangles()
            o3d_mesh_bg = o3d_mesh_bg.remove_duplicated_triangles()
            o3d_mesh_bg = o3d_mesh_bg.remove_duplicated_vertices()
            o3d_mesh_bg = o3d_mesh_bg.remove_non_manifold_edges()
            o3d_mesh_bg = o3d_mesh_bg.subdivide_loop(1)

            # combine
            o3d_mesh = o3d_mesh_fg + o3d_mesh_bg
            mesh_new = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                       faces=np.asarray(o3d_mesh.triangles))
            new_n_face = len(mesh_new.faces)
            print(f'subdividing from {len(faces)} to {new_n_face} faces')
            with open(f'{model_path}/mesh_subdivided.obj', 'w') as file:
                mesh_new.export(file_obj=file, file_type='obj')
            verts = torch.from_numpy(mesh_new.vertices).to(verts)
            faces = torch.from_numpy(mesh_new.faces).to(faces)
            fg_bg_nfaces = torch.tensor([len(o3d_mesh_fg.triangles), 
                                         len(o3d_mesh_bg.triangles)], device='cuda')
        
        if need_simplify_mesh:
            if fg_bg_nfaces[1]==0:
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
                o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(training_args.simplify_nface)
                o3d_mesh = o3d_mesh.remove_degenerate_triangles()
                o3d_mesh = o3d_mesh.remove_duplicated_triangles()
                o3d_mesh = o3d_mesh.remove_duplicated_vertices()
                o3d_mesh = o3d_mesh.remove_non_manifold_edges()
                mesh_new = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                        faces=np.asarray(o3d_mesh.triangles))
                
                print(f'simplified from {len(faces)} to {len(mesh_new.faces)} faces')
                with open(f'{model_path}/mesh_simplified.obj', 'w') as file:
                    mesh_new.export(file_obj=file, file_type='obj')
                verts = torch.from_numpy(mesh_new.vertices).to(verts)
                faces = torch.from_numpy(mesh_new.faces).to(faces)
                fg_bg_nfaces = torch.tensor([faces.shape[0], 0], device='cuda')
            else:
                fg_nf, bg_nf = fg_bg_nfaces.tolist()
                fg_nf_ = int(fg_nf/(fg_nf+bg_nf) * training_args.simplify_nface)
                bg_nf_ = training_args.simplify_nface - fg_nf_

                o3d_mesh_fg = o3d.geometry.TriangleMesh()
                o3d_mesh_fg.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
                o3d_mesh_fg.triangles = o3d.utility.Vector3iVector(faces[:fg_nf].detach().cpu().numpy())
                o3d_mesh_fg = o3d_mesh_fg.remove_unreferenced_vertices()
                o3d_mesh_fg = o3d_mesh_fg.simplify_quadric_decimation(fg_nf_)
                o3d_mesh_fg = o3d_mesh_fg.remove_degenerate_triangles()
                o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_triangles()
                o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_vertices()
                o3d_mesh_fg = o3d_mesh_fg.remove_non_manifold_edges()

                o3d_mesh_bg = o3d.geometry.TriangleMesh()
                o3d_mesh_bg.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
                o3d_mesh_bg.triangles = o3d.utility.Vector3iVector(faces[fg_nf:].detach().cpu().numpy())
                o3d_mesh_bg = o3d_mesh_bg.remove_unreferenced_vertices()
                o3d_mesh_bg = o3d_mesh_bg.simplify_quadric_decimation(bg_nf_)
                o3d_mesh_bg = o3d_mesh_bg.remove_degenerate_triangles()
                o3d_mesh_bg = o3d_mesh_bg.remove_duplicated_triangles()
                o3d_mesh_bg = o3d_mesh_bg.remove_duplicated_vertices()
                o3d_mesh_bg = o3d_mesh_bg.remove_non_manifold_edges()

                # combine
                o3d_mesh = o3d_mesh_fg + o3d_mesh_bg
                mesh_new = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                           faces=np.asarray(o3d_mesh.triangles))
                
                print(f'simplified from {len(faces)} to {len(mesh_new.faces)} faces')
                with open(f'{model_path}/mesh_simplified.obj', 'w') as file:
                    mesh_new.export(file_obj=file, file_type='obj')
                verts = torch.from_numpy(mesh_new.vertices).to(verts)
                faces = torch.from_numpy(mesh_new.faces).to(faces)
                fg_bg_nfaces = torch.tensor([len(o3d_mesh_fg.triangles), 
                                             len(o3d_mesh_bg.triangles)], device='cuda')

        if need_new_bind_rule:
            # bake texture to new Gaussians
            bc_coords, rad_base = mesh_utils.generate_barycentric_v2(training_args.s3_gs_per_face, 'cuda')

        if need_subdivision or need_new_bind_rule or need_simplify_mesh:
            print(f'bake texture to new Gaussians')
            n_c = 3*((self.active_sh_degree+1) ** 2)  # 3
            mlp_texture = MLPTexture3D(AABB, channels=n_c, min_max=None)
            mlp_texture.load_state_dict(mlp_texture_state)
            cov3D_L, features, opacity = \
                self.bind_gs_to_face(verts, faces, bc_coords, rad_base, 
                                    mlp_texture, self.active_sh_degree)

        n_face = faces.shape[0]
        N_gs_per_face = bc_coords.shape[0]
        N_gs = n_face * N_gs_per_face
        self.N_gs_per_face = N_gs_per_face
        self.fg_bg_nfaces = fg_bg_nfaces

        # -------------------------------------------------- #
        # init Gaussian center, scale, and rotation
        # -------------------------------------------------- #
        self.bc_coords = bc_coords
        self.rad_base = rad_base
        
        # effect of subdivision  (TODO: test if it's necessary to inherit scale factor)
        max_scale = 2.  # temporarily common, so hardcode this
        scale_factor = torch.tanh(scale_factor_fgbg) * max_scale
        sf_per_face = torch.repeat_interleave(scale_factor, fg_bg_nfaces)
        cov3D_L = cov3D_L.view(n_face, 3, 3).mul_(sf_per_face.view(n_face, 1, 1))
       
        # triangle-space cov to scale2D(s1,s2,eps) and rot2D(a+bi)
        self.thin_z_scale = torch.sqrt(cov3D_L[0, 2, 2]).item()
        cov_matrix = torch.bmm(cov3D_L[:, :2, :2],  # only consider x, y
                               cov3D_L[:, :2, :2].transpose(1, 2))
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        scale2D = torch.sqrt(eigenvalues.clamp_min(1e-30))
        scale2D = torch.log(scale2D)  # inverse activation (exp)

        # R_t2w = [[a, -b], [b, a]], first base is (a, b)
        rot2D = torch.nn.functional.normalize(eigenvectors[:, 0, :], dim=1)  # (a, b)

        # shared by face gaussians
        scale2D = scale2D.view(n_face, 1, 2).repeat(1, N_gs_per_face, 1).view(N_gs, 2).detach().clone()
        rot2D = rot2D.view(n_face, 1, 2).repeat(1, N_gs_per_face, 1).view(N_gs, 2).detach().clone()

        # -------------------------------------------------- #
        # init Gaussian SH and opacity
        # -------------------------------------------------- #
        _features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        _features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        _opacity = inverse_sigmoid(torch.full_like(opacity, training_args.init_opacity))

        # -------------------------------------------------- #
        # construct parameters and config optimizer
        # -------------------------------------------------- #
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scale2D.requires_grad_(True))
        self._rotation = nn.Parameter(rot2D.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        
        self.verts = nn.Parameter(verts.requires_grad_(True))
        # self.initial_verts = verts.detach().clone()
        self.faces = faces

        self.training_setup(training_args)


    def clear_space(self):
        del self.rot_t2w
        del self._xyz

    def renew_gaussian(self, viewpoint_cam=None):
        # ------------------------ #
        # verts -> gaussians
        # ------------------------ #
        verts, faces = self.verts, self.faces
        if not self.eval:
            self.clear_space()
        self.face_mask = None
        self.gs_mask = None
        if self.use_frustum and (viewpoint_cam is not None):
            query_p = verts[faces].mean(dim=1)
            self.face_mask = in_frustum(viewpoint_cam.full_proj_transform.cuda(), query_p)
            self.gs_mask = torch.repeat_interleave(self.face_mask, self.N_gs_per_face)
            faces = faces[self.face_mask]
        if self.eval:  # only need gs_mask for eval.
            return

        # with torch.no_grad():
        face_normals = mesh_utils.face_normals(verts, faces, unit=True)
        face_x = torch.nn.functional.normalize(verts[faces[:, 1]] - verts[faces[:, 0]])
        face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
        self.rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)

        gs_xyz = torch.matmul(  # [n_triangle, n_inner_gs, 3] @ [n_triangle, 3, 3]
            self.bc_coords.view(1, -1, 3), verts[faces])
        self._xyz = gs_xyz.view(-1, 3)
    
    def reg_loss(self, w_laplace=0.1, w_normal_consistency=0.1):
        # TODO: compare whether to use face mask
        curr_face = self.faces[self.face_mask] if (self.face_mask is not None) else self.faces
        mesh = Meshes(verts=[self.verts], faces=[curr_face], textures=None)
        loss = 0.
        if w_laplace > 0:
            loss += w_laplace * mesh_laplacian_smoothing(mesh, method='uniform')
        if w_normal_consistency > 0:
            loss += w_normal_consistency * mesh_normal_consistency(mesh)
        return loss
    
    @torch.no_grad()
    def export_mesh(self, path):
        mesh = trimesh.Trimesh(vertices=self.verts.detach().cpu().numpy(), 
                               faces=self.faces.detach().cpu().numpy())
        with open(path, 'w') as file:
            mesh.export(file_obj=file, file_type='obj')


    def restore(self, model_args, training_args):
        raise NotImplementedError

    @property
    def get_scaling(self):
        _scaling = self._scaling[self.gs_mask] if (self.gs_mask is not None) else self._scaling
        _scaling3D = torch.cat([
            self.scaling_activation(_scaling),  # 2D component
            torch.full((_scaling.shape[0], 1), self.thin_z_scale, device='cuda')
        ], dim=1)

        return _scaling3D
    
    @property
    def get_rotation(self):
        if self.eval:
            rot = self._rotation[self.gs_mask] if (self.gs_mask is not None) else self._rotation
            return rot  # already 3D via `load_for_eval()`, and activated
        R = self.get_rot_matrix()
        _rotation = matrix_to_quaternion(R)

        return self.rotation_activation(_rotation)
    
    def get_rot_matrix(self):
        _rotation = self._rotation[self.gs_mask] if (self.gs_mask is not None) else self._rotation
        # triangle space R_t = a -b  0
        #                      b  a  0
        #                      0  0  1 
        R_t = torch.zeros((_rotation.shape[0], 3, 3), device='cuda')
        c = torch.nn.functional.normalize(_rotation, dim=1)  # a+bi
        R_t[:, 0, 0] =  c[:, 0]
        R_t[:, 0, 1] = -c[:, 1]
        R_t[:, 1, 0] =  c[:, 1]
        R_t[:, 1, 1] =  c[:, 0]
        R_t[:, 2, 2] =  1.0
        # world space R
        n_triangle = self.rot_t2w.shape[0]
        R_t2w = self.rot_t2w.view(n_triangle, 1, 3, 3)
        R_t = R_t.view(n_triangle, -1, 3, 3)
        R = torch.matmul(R_t2w, R_t).view(-1, 3, 3)
        return R

    @property
    def get_xyz(self):
        if self.eval:
            return self._xyz[self.gs_mask] if (self.gs_mask is not None) else self._xyz
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc[self.gs_mask] if (self.gs_mask is not None) else self._features_dc
        features_rest = self._features_rest[self.gs_mask] if (self.gs_mask is not None) else self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        _opacity = self._opacity[self.gs_mask] if (self.gs_mask is not None) else self._opacity
        return self.opacity_activation(_opacity)
    
    def get_covariance(self, scaling_modifier=1):
        assert scaling_modifier==1, "not supported"
        _s = self.get_scaling
        S = torch.zeros((_s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        S[:, 0, 0] = _s[:, 0]
        S[:, 1, 1] = _s[:, 1]
        S[:, 2, 2] = _s[:, 2]

        R = self.get_rot_matrix()
        
        # Sigma = RS(RS)^T
        L = R @ S
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)

        return symm

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # the only used term
        self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, training_args):
        # spatial_lr_scale = 10. * bbox_radius / torch.tensor(n_vertices_in_fg).pow(1/2).item()
        l = [
            {'params': [self.verts], 'lr': training_args.vert_lr, "name": "verts"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if training_args.vert_lr_final is None:
            training_args.vert_lr_final = training_args.vert_lr
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.vert_lr,
                                                    lr_final=training_args.vert_lr_final,
                                                    max_steps=5000)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "verts":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def reset_opacity(self, training_args):
        opacities_new = inverse_sigmoid(torch.full_like(self._opacity, training_args.init_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
