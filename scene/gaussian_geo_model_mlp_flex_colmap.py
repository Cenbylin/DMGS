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
from utils.effi_utils import count_time

import trimesh
import point_cloud_utils as pcu
from geo.flexicubes_no_adjacent import FlexiCubes

import geo.mesh_utils as mesh_utils
from geo.texture import MLPTexture3D
import torch.nn.functional as F

@torch.no_grad()
@torch.jit.script
def in_frustum(proj_matrix: torch.Tensor, verts: torch.Tensor, cube_len: float, piece_id: int, n_piece: int):
    # following in_frustum() in cuda_rasterize
    # method-1 use projection matrix.
    proj_matrix_3x3 = proj_matrix[:3, :]
    # trick: we fuse the `add cube_len/2. to verts` into the proj_matrix, to save time&memory
    # so, below = (vert + cube_len/2.) @ proj_matrix_3x3 + proj_matrix_1x3
    proj_matrix_1x3 = proj_matrix[3:, :] + (proj_matrix_3x3*(cube_len/2.)).sum(dim=0, keepdim=True)
    p_proj = torch.matmul(verts, proj_matrix_3x3)
    p_proj += proj_matrix_1x3  #  # [N, 4] @ [4, 4], use right-mul
    w = (p_proj[:, 3:] + 1e-6)
    p_proj = p_proj[:, :3] / w
    if piece_id<0:
        mask = (p_proj.abs()<1.05).all(dim=-1)  # within screen space
    elif n_piece==2:
        if piece_id==0:
            mask = (p_proj > torch.tensor([[-1., -1., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 0., 1.,  1.]], device='cuda')+0.05).all(dim=-1)  # within 1/2 screen space
        elif piece_id==1:
            mask = (p_proj > torch.tensor([[ 0., -1., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 1., 1., 1.]], device='cuda')+0.05).all(dim=-1)  # within 1/2 screen space
        else:
            raise NotImplementedError
    elif n_piece==4:
        if piece_id==0:
            mask = (p_proj > torch.tensor([[-1., -1., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 0.,  0.,  1.]], device='cuda')+0.05).all(dim=-1)  # within 1/4 screen space
        elif piece_id==1:
            mask = (p_proj > torch.tensor([[ 0., -1., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 1.,  0.,  1.]], device='cuda')+0.05).all(dim=-1)  # within 1/4 screen space
        elif piece_id==2:
            mask = (p_proj > torch.tensor([[-1.,  0., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 0.,  1.,  1.]], device='cuda')+0.05).all(dim=-1)  # within 1/4 screen space
        elif piece_id==3:
            mask = (p_proj > torch.tensor([[ 0.,  0., -1.]], device='cuda')-0.05).all(dim=-1) \
                    & (p_proj < torch.tensor([[ 1.,  1.,  1.]], device='cuda')+0.05).all(dim=-1)  # within 1/4 screen space
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    mask &= (w.squeeze()>0)  # in front of camera

    return mask

class GaussianGeoModel(nn.Module):

    def __init__(self, sh_degree : int, gs_per_face, c2f_rate):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)  # dynamically generated from scale2D
        self._rotation = torch.empty(0)  # dynamically generated from rot2D
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # added tmp var
        self.cov3D_L = torch.empty(0)
        self.verts = torch.empty(0)
        self.faces = torch.empty(0)
        self.sdf_reg_loss = torch.empty(0)
        self.flexi_reg_loss = torch.empty(0)
        self.rot_t2w = torch.empty(0)
        self.max_frame_nface = 0
        self.c2f_rate = c2f_rate

        # added
        self.sdf = None
        self.deform = None
        self.last_sdf = None  # for tracking topology changes
        self.bc_coords, self.rad_base = mesh_utils.generate_barycentric_v2(gs_per_face, 'cuda')
        self.rot_t2w = None

    def capture(self, gs_info):
        return (
            # last matched result
            gs_info['xyz'],
            gs_info['features'],
            gs_info['opacity'],  # activated
            gs_info['cov3D_L'],  # haven't multiplied scale_factor
            gs_info['verts'], 
            gs_info['faces'],
            gs_info['fg_bg_nfaces'],

            # model states
            self.optimizer.state_dict(),
            self.active_sh_degree,
            self.spatial_lr_scale,
            self.bc_coords,
            self.rad_base,
            torch.cat([self.scale_factor_fg, self.scale_factor_bg]),
            self.mlp_texture.AABB,
            self.mlp_texture.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        raise NotImplementedError  # paired with capture()


    def init_geo(self, coarse_mesh, aabb_fg_bg, ncube_fg, ncube_bg=None):
        device = "cuda"
        # extrator
        self.marching_geo    = FlexiCubes()
        # aabb
        aabb_fg = torch.tensor(aabb_fg_bg[0], device=device).view(2, 3)
        aabb_bg = torch.tensor(aabb_fg_bg[1], device=device).view(2, 3) if aabb_fg_bg[1] else None
        self.aabb_fg, self.aabb_bg = aabb_fg, aabb_bg
        # coarse mesh info
        vw, fw = coarse_mesh.vertices.astype(np.float32), coarse_mesh.faces
        
        #####################################
        # foreground
        #####################################
        # 1. grid
        (self.grid_res, self.marching_verts, self.cube_edge_fg,
         self.max_displacement) = self.setup_grid(aabb_fg, ncube_fg)
        self.cube_ind_cpu = self.marching_geo.construct_cubes(
                torch.ones_like(self.marching_verts[:, 0], dtype=torch.bool), self.grid_res).to('cpu')
        # 2. Learnable parts
        sdf, fid, bc = pcu.signed_distance_to_mesh(self.marching_verts.cpu().numpy(), vw, fw)
        self.sdf = torch.nn.Parameter(torch.tensor(sdf, device=device), requires_grad=True)
        self.deform_fg = torch.nn.Parameter(torch.zeros_like(self.marching_verts), requires_grad=True) \
                if (ncube_fg == self.target_ncube_fg) else None  # only final fine-grid can deform
        
        self.max_scale = 2.
        self.scale_factor_fg = torch.nn.Parameter(
            torch.arctanh(torch.full((1,), 1/self.max_scale, device=device)), requires_grad=True)
        self.scale_factor_bg = torch.nn.Parameter(
            torch.arctanh(torch.full((1,), 1/self.max_scale, device=device)), requires_grad=True)
        torch.cuda.empty_cache()

        #####################################
        # [added] background
        #####################################
        if self.enable_bg:
            # 1. grid
            (self.grid_res_bg, self.marching_verts_bg, self.cube_edge_bg,
             self.max_displacement_bg) = self.setup_grid(aabb_bg, ncube_bg)
            
            # added [skip fg part]
            l = (aabb_fg[1]-aabb_fg[0])*0.05  #  |aabb| to ->l|tighter_aabb|l<-
            self.marching_verts_mask = (self.marching_verts_bg < (aabb_fg[0]+l).view(1,3)).any(dim=1) \
                                     | (self.marching_verts_bg > (aabb_fg[1]-l).view(1,3)).any(dim=1)
            self.cube_ind_cpu_bg = self.marching_geo.construct_cubes(
                self.marching_verts_mask, self.grid_res_bg).to('cpu')

            # 4. Learnable parts
            sdf_bg, fid, bc = pcu.signed_distance_to_mesh(self.marching_verts_bg.cpu().numpy(), vw, fw)
            self.sdf_bg = torch.nn.Parameter(torch.tensor(sdf_bg, device=device), requires_grad=True)
            self.deform_bg = torch.nn.Parameter(torch.zeros_like(self.marching_verts_bg), requires_grad=True) \
                if (ncube_bg == self.target_ncube_bg) else None  # only final fine-grid can deform

            torch.cuda.empty_cache()

    def get_cube_n(self, target_ncube, n_c2f_steps, curr_step=-1):
        rest_res = n_c2f_steps - (curr_step+1)
        return round(target_ncube/(self.c2f_rate**rest_res))
    
    def coarse_to_fine(self, n_c2f_steps, curr_step, opt):
        del self.marching_verts, self.cube_edge_fg
        # foreground
        ncube_fg = self.get_cube_n(self.target_ncube_fg, n_c2f_steps, curr_step)
        (new_grid_res, self.marching_verts, self.cube_edge_fg,
         self.max_displacement) = self.setup_grid(self.aabb_fg, ncube_fg)
        self.cube_ind_cpu = self.marching_geo.construct_cubes(
                torch.ones_like(self.marching_verts[:, 0], dtype=torch.bool), new_grid_res).to('cpu')
        self.sdf = torch.nn.Parameter(  # inherit information
            F.interpolate(self.sdf.data.view(tuple([1, 1]+[r+1 for r in self.grid_res])), 
                size=tuple([r+1 for r in new_grid_res]), mode='trilinear', align_corners=True).flatten(),
            requires_grad=True)
        self.deform = torch.nn.Parameter(torch.zeros_like(self.marching_verts), requires_grad=True) \
                if (ncube_fg == self.target_ncube_fg) else None  # only final fine-grid can deform
        self.grid_res = new_grid_res

        # # reset scale, may avoid: 1) foreground occlusion issue; 2) local optimum
        # self.scale_factor_fg = torch.nn.Parameter(
        #     torch.arctanh(torch.full((1,), 1/self.max_scale, device='cuda')), requires_grad=True)
        # self.scale_factor_bg = torch.nn.Parameter(
        #     torch.arctanh(torch.full((1,), 1/self.max_scale, device='cuda')), requires_grad=True)

        if self.enable_bg:
            del self.marching_verts_bg, self.cube_edge_bg
            ncube_bg = self.get_cube_n(self.target_ncube_bg, n_c2f_steps, curr_step)
            (new_grid_res_bg, self.marching_verts_bg, self.cube_edge_bg,
            self.max_displacement_bg) = self.setup_grid(self.aabb_bg, ncube_bg)
            
            # added [skip fg part]
            l = (self.aabb_fg[1]-self.aabb_fg[0])*0.05  #  |aabb| to ->l|tighter_aabb|l<-
            self.marching_verts_mask = (self.marching_verts_bg < (self.aabb_fg[0]+l).view(1,3)).any(dim=1) \
                                     | (self.marching_verts_bg > (self.aabb_fg[1]-l).view(1,3)).any(dim=1)
            self.cube_ind_cpu_bg = self.marching_geo.construct_cubes(
                self.marching_verts_mask, new_grid_res_bg).to('cpu')
            
            self.sdf_bg = torch.nn.Parameter(  # inherit information
                F.interpolate(self.sdf_bg.data.view([1, 1]+[r+1 for r in self.grid_res_bg]), 
                    size=tuple([r+1 for r in new_grid_res_bg]), mode='trilinear', align_corners=True).flatten(),
                requires_grad=True)
            self.deform_fg = torch.nn.Parameter(torch.zeros_like(self.marching_verts_bg), requires_grad=True) \
                    if (ncube_bg == self.target_ncube_bg) else None  # only final fine-grid can deform
            self.grid_res_bg = new_grid_res_bg

        self.training_setup(opt)

    def setup_grid(self, aabb, ncube):
        # 1. xyz range and resolutions
        aabb_edge = (aabb[1]-aabb[0])
        cube_edge = float((aabb_edge.prod() / (ncube)) ** (1/3))
        res_fg_list = (aabb_edge / cube_edge).int().tolist()
        print("Foreground grid:", res_fg_list)

        # 2. init flexicube (differentiable iso-surface extrator)
        grid_res = res_fg_list
        marching_verts = self.marching_geo.construct_voxel_grid_vert_only(
            grid_res, aabb[0], aabb[1])
        max_displacement = (cube_edge / 4.)

        return grid_res, marching_verts, cube_edge, max_displacement

    def create_from_mesh(self, training_args, coarse_mesh_path, 
                         aabb_fg_bg, model_path):
        # 1. load coarse mesh
        coarse_mesh = trimesh.load_mesh(coarse_mesh_path)
        
        # 2. init geometry(sdf, deform)
        self.enable_bg = (training_args.res_bg>0)
        n_c2f_steps = len(training_args.c2f_steps)
        self.target_ncube_fg = training_args.res_fg**3
        self.target_ncube_bg = training_args.res_bg**3
        print(f"Begin to initialize Geometry, enable_bg: {self.enable_bg}.")
        self.init_geo(
            # init sdf (watertight mesh)
            coarse_mesh=coarse_mesh, aabb_fg_bg=aabb_fg_bg,
            # [foreground] differentiable geometry
            ncube_fg=self.get_cube_n(self.target_ncube_fg, n_c2f_steps, -1),
            # [foreground] differentiable geometry
            ncube_bg=self.get_cube_n(self.target_ncube_bg, n_c2f_steps, -1)
        )

        # 3. texture
        largest_aabb = aabb_fg_bg[1] if self.enable_bg else aabb_fg_bg[0]
        AABB = torch.tensor(largest_aabb, device='cuda')
        assert AABB.shape == (2, 3)
        n_c = 3 * ((self.max_sh_degree + 1) ** 2)
        self.mlp_texture = MLPTexture3D(AABB, channels=n_c, min_max=None)

        self.training_setup(training_args)
        self.warm_up = False
        self.train_fg = True
        self.train_bg = True

        # for debugging
        os.makedirs(model_path, exist_ok=True)
        self.export_mesh(os.path.join(model_path, "init_mesh.obj"))

        # for ablation study
        self.adaptive_cov = training_args.adaptive_cov
        print("Adaptive Covariance:", self.adaptive_cov)

    def getMesh_in_frustum(self, viewpoint_cam, piece_id, n_piece, training):
        # make all out_furstum sdf be 1 (>0). they will be filtered during marching.
        vert_mask = in_frustum(viewpoint_cam.full_proj_transform.cuda(), 
                               self.marching_verts, self.cube_edge_fg, piece_id, n_piece)
        indices = self.marching_geo.construct_cubes(vert_mask, self.grid_res)
        # debug
        # from geo.mesh_utils import numpy_to_ply
        # numpy_to_ply("in_frustum_v.ply", self.marching_verts[vert_mask].detach().cpu().numpy())
        with torch.set_grad_enabled(self.train_fg):
            if self.deform is None:
                v_deformed = self.marching_verts
            else:
                v_deformed = self.marching_verts + self.max_displacement * torch.tanh(self.deform)
            verts, faces, reg_loss, surf_edges = self.marching_geo(v_deformed, self.sdf, indices, self.grid_res, 
                                # self.per_cube_weights[:,:12], self.per_cube_weights[:,12:20], self.per_cube_weights[:,20],
                                training=(self.train_fg and training))
            n_vert_fg = verts.shape[0]
            n_face_fg = faces.shape[0]
        need_loss_fg = (self.train_fg and (surf_edges is not None))
        flexi_reg_loss = reg_loss.mean() if need_loss_fg else 0.
        sdf_reg_loss = self.compute_sdf_reg_loss(self.sdf, surf_edges) if need_loss_fg else 0.

        if self.enable_bg:
            # make all out_furstum sdf be 1 (>0). they will be filtered during marching.
            vert_mask = in_frustum(viewpoint_cam.full_proj_transform.cuda(), 
                                   self.marching_verts_bg, self.cube_edge_bg, piece_id, n_piece)
            vert_mask &= self.marching_verts_mask
            indices_bg = self.marching_geo.construct_cubes(vert_mask, self.grid_res_bg)
            with torch.set_grad_enabled(self.train_bg):
                if self.deform_bg is None:
                    v_deformed = self.marching_verts_bg
                else:
                    v_deformed = self.marching_verts_bg + self.max_displacement_bg * torch.tanh(self.deform_bg)
                verts_bg, faces_bg, reg_loss, surf_edges = self.marching_geo(v_deformed, self.sdf_bg, indices_bg, self.grid_res_bg, 
                                # self.per_cube_weights_bg[:,:12], self.per_cube_weights_bg[:,12:20], self.per_cube_weights_bg[:,20],
                                training=training)
            if surf_edges is not None:  # might be no face
                # print('warnning: background march 0 faces')
                flexi_reg_loss += reg_loss.mean()
                sdf_reg_loss += self.compute_sdf_reg_loss(self.sdf_bg, surf_edges)

                verts = torch.cat([verts, verts_bg], dim=0)
                faces = torch.cat([faces, faces_bg+n_vert_fg], dim=0)
        
        # applications like: applying different scale.
        fg_bg_nfaces = torch.tensor([n_face_fg, faces.shape[0]-n_face_fg], device=verts.device)
        return verts, faces, fg_bg_nfaces, flexi_reg_loss, sdf_reg_loss

    @torch.no_grad()
    def getMesh_full(self):
        if self.deform is None:
            v_deformed = self.marching_verts
        else:
            v_deformed = self.marching_verts + self.max_displacement * torch.tanh(self.deform)
        verts, faces, reg_loss, surf_edges = self.marching_geo(
            v_deformed, self.sdf, self.cube_ind_cpu.cuda(), self.grid_res, training=False, de_ambiguity=False)

        n_vert_fg = verts.shape[0]
        n_face_fg = faces.shape[0]
        # self.flexi_reg_loss = reg_loss.mean()

        if self.enable_bg:
            if self.deform_bg is None:
                v_deformed = self.marching_verts_bg
            else:
                v_deformed = self.marching_verts_bg + self.max_displacement * torch.tanh(self.deform_bg)
            verts_bg, faces_bg, reg_loss, surf_edges = self.marching_geo(
                v_deformed, self.sdf_bg, self.cube_ind_cpu_bg.cuda(), self.grid_res_bg, training=False)
            # self.flexi_reg_loss += reg_loss.mean()

            verts = torch.cat([verts, verts_bg], dim=0)
            faces = torch.cat([faces, faces_bg+n_vert_fg], dim=0)
        
        # applications like: applying different scale.
        fg_bg_nfaces = torch.tensor([n_face_fg, faces.shape[0]-n_face_fg], device=verts.device)
        return verts, faces, fg_bg_nfaces, None, None
        
    def renew_gaussian(self, train_mesh, viewpoint_cam=None, piece_id=-1, n_piece=0):
        # -------------------------------------------------- #
        # 1. sdf/deform -> verts -> gaussians
        # 2. sample gaussian colors in 3D-texture
        # -------------------------------------------------- #
        device = 'cuda'
        if viewpoint_cam is not None:
            verts, faces, fg_bg_nfaces, flexi_reg_loss, sdf_reg_loss = \
                self.getMesh_in_frustum(viewpoint_cam, piece_id, n_piece, train_mesh)
            self.max_frame_nface = max(self.max_frame_nface, int(faces.shape[0]))
        else:
            verts, faces, fg_bg_nfaces, flexi_reg_loss, sdf_reg_loss = \
                self.getMesh_full()  # no_grad

        # triangle space definition & data shape
        face_vert = verts[faces]
        v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
        v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
        face_x = torch.div(v1v2, v1v2_len)
        face_normals = mesh_utils.face_normals(verts, faces, unit=True)
        face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
        rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
        n_face = faces.shape[0]
        N_gs_per_face = self.bc_coords.shape[0]
        N_gs = n_face * N_gs_per_face
        
        # gaussian means
        gs_xyz = torch.matmul(
            self.bc_coords.view(1, -1, 3), face_vert)  # [1, n_inner_gs, 3] @ [n_triangle, 3, 3]
        gs_xyz = gs_xyz.view(-1, 3)  # [n_triangle*n_inner_gs, 3]
        
        # gaussian covariance (triangle space)
        with torch.no_grad():
            v1v2_xlen = v1v2_len.view(n_face)
            v1v3 = face_vert[:, 2, :] - face_vert[:, 0, :]
            
            if self.adaptive_cov:
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
            s_scalar = (v1v2_xlen * self.rad_base)  # s_scalar is std-deviation
            L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_face, 1, 1]

            cov3D_L = torch.zeros((n_face, 3, 3), device=device)
            cov3D_L[:, :2, :2] = L.view(n_face, 2, 2)
            cov3D_L[:, 2, 2] = self.spatial_lr_scale * 1e-6  # flat

        # gaussian colors & opacities
        SH_dim = (self.max_sh_degree+1) ** 2
        N_gs_fg = fg_bg_nfaces[0] * N_gs_per_face
        if self.warm_up:  # fix foreground opacity(0.5), color(SH=0)
            opacities = torch.cat([
                torch.full((N_gs_fg, 1), 0.5, device=device),
                torch.full((N_gs-N_gs_fg, 1), 0.9999, device=device)
            ], dim=0)
            features = torch.cat([
                torch.full((N_gs_fg, 3, SH_dim), 0., device=device),
                self.mlp_texture.sample_noact(gs_xyz[N_gs_fg:]).view(-1, 3, SH_dim)
            ], dim=0)
        else:
            opacities = torch.full((N_gs, 1), 0.9999, device=device)
            if self.train_fg and self.train_bg:
                features = self.mlp_texture.sample_noact(gs_xyz).view(N_gs, 3, SH_dim)
            else:  # carefully track gradient
                with torch.set_grad_enabled(self.train_fg):
                    features_fg = self.mlp_texture.sample_noact(gs_xyz[:N_gs_fg]).view(-1, 3, SH_dim)
                with torch.set_grad_enabled(self.train_bg):
                    features_bg = self.mlp_texture.sample_noact(gs_xyz[N_gs_fg:]).view(-1, 3, SH_dim)
                features = torch.cat([features_fg, features_bg], dim=0)
        
        # -------------------------------------------------- #
        # return dynamically generated gaussian information
        # -------------------------------------------------- #
        gs_info = {
            'xyz': gs_xyz,
            'opacity': opacities,
            'covariance': self.get_covariance_dyn(rot_t2w, cov3D_L, fg_bg_nfaces),
            'cov3D_L': cov3D_L,
            'features': features,
            'active_sh_degree': self.active_sh_degree,
            'max_sh_degree': self.max_sh_degree,
            'flexi_reg_loss': flexi_reg_loss,
            'sdf_reg_loss': sdf_reg_loss,
            'verts': verts,
            'faces': faces,
            'fg_bg_nfaces': fg_bg_nfaces,
        }
        return gs_info

    def compute_sdf_reg_loss(self, sdf, edges):
        sdf_f1x8x2 = sdf[edges.reshape(-1)].reshape(-1,2)
        mask = torch.sign(sdf_f1x8x2[...,0]) != torch.sign(sdf_f1x8x2[...,1])
        sdf_f1x8x2 = sdf_f1x8x2[mask]
        sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,0], (sdf_f1x8x2[...,1] > 0).float()) + \
                torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,1], (sdf_f1x8x2[...,0] > 0).float())
        return sdf_diff

    @torch.no_grad()
    def export_mesh(self, path):
        verts, faces, _, _, _ = self.getMesh_full()
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                               faces=faces.detach().cpu().numpy())
        with open(path, 'w') as file:
            mesh.export(file_obj=file, file_type='obj')
        torch.cuda.empty_cache()
 
    def get_covariance_dyn(self, rot_t2w, cov3D_L, fg_bg_nfaces, scaling_modifier=1):
        assert scaling_modifier==1, "not supported"

        n_triangle = rot_t2w.shape[0]
        n_inner_gs = self.bc_coords.shape[0]

        # per triangle operations (2.23 disabled learnable global scale)
        # sf_fg = (self.scale_factor_fg.detach() if self.warm_up else self.scale_factor_fg)
        # scale_factor = torch.tanh(torch.cat([sf_fg, self.scale_factor_bg])) * self.max_scale
        # sf_per_face = torch.repeat_interleave(scale_factor, fg_bg_nfaces)
        L = cov3D_L.view(n_triangle, 3, 3)  # * sf_per_face.view(n_triangle, 1, 1)
        R_t2w = rot_t2w.view(n_triangle, 3, 3)  # world space R
        L = torch.matmul(R_t2w, L).view(-1, 3, 3)  # [n_triangle,3,3]
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)

        # shared by all inner gs
        symm = symm.view(n_triangle, 1, 6).expand(-1, n_inner_gs, -1).reshape(-1, 6)
        return symm
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # the only used term
        self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, opt):
        sdf_lr = opt.sdf_lr[0] if isinstance(opt.sdf_lr, list) else opt.sdf_lr
        sdf_lr_bg = opt.sdf_lr_bg[0] if isinstance(opt.sdf_lr_bg, list) else opt.sdf_lr_bg

        l = [
            {'params': list(self.mlp_texture.parameters()), 'lr': opt.texture_lr, "name": "texture"},
            {'params': [self.sdf], 'lr': sdf_lr, "name": "sdf"},
            # {'params': [self.scale_factor_fg], 'lr': opt.scale_factor_lr, "name": "scale_factor_fg"},
        ]
        if self.deform is not None:
            l += [{'params': [self.deform], 'lr': opt.deform_lr, "name": "deform"}]
        
        if self.enable_bg:
            l += [
                {'params': [self.sdf_bg], 'lr': sdf_lr_bg, "name": "sdf_bg"},
                {'params': [self.scale_factor_bg], 'lr': opt.scale_factor_lr, "name": "scale_factor_bg"},
            ]
            if self.deform_bg is not None:
                l += [{'params': [self.deform_bg], 'lr': opt.deform_lr_bg, "name": "deform_bg"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.init_lr_collector = {item['name']:item['lr'] for item in l}
        self.scheduler_args = get_expon_lr_func(
            lr_init=1.0, lr_final=opt.final_lr_rate, max_steps=opt.reg_decay_iter)

    def update_learning_rate(self, iteration):
        lr_decay_rate = self.scheduler_args(iteration)
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["sdf", "sdf_bg"]:
                param_group['lr'] = lr_decay_rate * self.init_lr_collector[param_group["name"]]

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

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
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
    
    def save_for_debug(self):
        attr_to_save = ['verts', 'max_displacement', 'deform', 'sdf', 'indices', 'grid_res', 'per_cube_weights', 'bc_coords', 'rad_base', 'spatial_lr_scale', 'max_sh_degree',]
        saving_dict = {}
        for k in attr_to_save:
            saving_dict[k] = self.__getattribute__(k)
        saving_dict['mlp_texture'] = self.mlp_texture.state_dict()
        torch.save(saving_dict, 'debug.pth')
