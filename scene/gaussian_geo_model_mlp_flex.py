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
from geo.flexicubes import FlexiCubes

import geo.mesh_utils as mesh_utils
from geo.texture import MLPTexture3D
from torch_scatter import scatter_add
import torch.nn.functional as F
import open3d as o3d

class GaussianGeoModel:

    def __init__(self, sh_degree : int, gs_per_face, c2f_rate):
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

        # added
        self.sdf = None
        self.deform = None
        self.last_sdf = None  # for tracking topology changes
        self.bc_coords, self.rad_base = mesh_utils.generate_barycentric_v2(gs_per_face, 'cuda')
        self.rot_t2w = None
        self.c2f_rate = c2f_rate

    def load_full(self, opt, model_args):
        # 1. mesh & gaussians info
        (xyz, features, opacity, cov3D_L, verts, faces, fg_bg_nfaces, 
         opt_dict, self.active_sh_degree, self.spatial_lr_scale, self.bc_coords, 
         self.rad_base, scale_factor_fgbg, AABB, mlp_texture_state,
         # extra states when `full_state` mode
         self.marching_verts, self.indices, self.sdf, self.deform, 
         self.per_cube_weights, self.max_displacement, self.grid_res,
         self.scale_factor, self.max_scale) = model_args

        self.marching_geo    = FlexiCubes()
        
        # recover parameters
        self.sdf = torch.nn.Parameter(self.sdf.requires_grad_(True))
        self.scale_factor = torch.nn.Parameter(self.scale_factor.requires_grad_(True))
        
        n_c = 3*((self.active_sh_degree+1) ** 2)
        self.mlp_texture = MLPTexture3D(AABB, channels=n_c, min_max=None)
        self.mlp_texture.load_state_dict(mlp_texture_state)

        if self.deform is not None:
            self.deform = torch.nn.Parameter(self.deform.requires_grad_(True))
            self.per_cube_weights = torch.nn.Parameter(self.per_cube_weights.requires_grad_(True))

        # recover trainer
        self.training_setup(opt)

        # for ablation study 
        self.adaptive_cov = opt.adaptive_cov
        

    def capture(self, gs_info, full_state=False):
        # model states
        model_states = (
            self.optimizer.state_dict(),
            self.active_sh_degree,
            self.spatial_lr_scale,
            self.bc_coords,
            self.rad_base,
            torch.cat([self.scale_factor, self.scale_factor]),
            self.mlp_texture.AABB,
            self.mlp_texture.state_dict())
        
        if full_state:
            model_states += (
                self.marching_verts, self.indices, self.sdf, self.deform, 
                self.per_cube_weights, self.max_displacement, self.grid_res,
                self.scale_factor, self.max_scale
            )

        return (
            # last matched result
            gs_info['xyz'],
            gs_info['features'],
            gs_info['opacity'],  # activated
            gs_info['cov3D_L'],  # haven't multiplied scale_factor
            gs_info['verts'], 
            gs_info['faces'],
            gs_info['fg_bg_nfaces'],
        ) + model_states
    
    def restore(self, model_args, training_args):
        raise NotImplementedError  # paired with capture()
    
    def get_cube_n(self, target_ncube, n_c2f_steps, curr_step=-1):
        rest_res = n_c2f_steps - (curr_step+1)
        return round(target_ncube/(self.c2f_rate**rest_res))
    
    def coarse_to_fine(self, n_c2f_steps, curr_step, opt):
        del self.marching_verts, self.indices
        # foreground
        ncube_fg = self.get_cube_n(self.target_ncube_fg, n_c2f_steps, curr_step)
        aabb_edge_fg = (self.aabb_fg[1]-self.aabb_fg[0])
        cube_edge_fg = (aabb_edge_fg.prod() / (ncube_fg)) ** (1/3)
        new_grid_res = (aabb_edge_fg / cube_edge_fg).int().tolist()
        self.marching_verts, self.indices = self.marching_geo.construct_voxel_grid_v2(
            new_grid_res, self.aabb_fg[0], self.aabb_fg[1])
        self.max_displacement = (cube_edge_fg / 4.)
        
        self.sdf = torch.nn.Parameter(  # inherit information
            F.interpolate(self.sdf.data.view(tuple([1, 1]+[r+1 for r in self.grid_res])), 
                size=tuple([r+1 for r in new_grid_res]), mode='trilinear', align_corners=True).flatten(),
            requires_grad=True)
        
        if (ncube_fg == self.target_ncube_fg):   # only final fine-grid can deform & use reg weight
            self.deform = torch.nn.Parameter(torch.zeros_like(self.marching_verts), requires_grad=True)
            self.per_cube_weights = torch.nn.Parameter(
                torch.ones((self.indices.shape[0], 21), dtype=torch.float32, device='cuda'), requires_grad=True)
        else:
            self.deform, self.per_cube_weights = None, None
        
        self.grid_res = new_grid_res
        print("new grid:", new_grid_res)
        self.training_setup(opt)

    def init_geo(self, coarse_mesh, aabb_fg, ncube_fg):
        device = "cuda"
        self.aabb_fg = aabb_fg
        # 1. coarse mesh info
        vw, fw = coarse_mesh.vertices.astype(np.float32), coarse_mesh.faces
        assert aabb_fg.shape == (2, 3)

        # 1. xyz range and resolutions
        aabb_edge_fg = (aabb_fg[1]-aabb_fg[0])
        cube_edge_fg = (aabb_edge_fg.prod() / (ncube_fg)) ** (1/3)
        res_fg_list = (aabb_edge_fg / cube_edge_fg).int().tolist()
        print("Init grid:", res_fg_list)

        # 2. init flexicube (differentiable iso-surface extrator)
        self.marching_geo    = FlexiCubes()
        self.grid_res = res_fg_list
        self.marching_verts, self.indices = self.marching_geo.construct_voxel_grid_v2(
            self.grid_res, aabb_fg[0], aabb_fg[1])
        self.max_displacement = (cube_edge_fg / 4.)

        # 3. Learnable parts
        sdf, fid, bc = pcu.signed_distance_to_mesh(self.marching_verts.cpu().numpy(), vw.astype(np.float32), fw)
        self.sdf = torch.nn.Parameter(torch.tensor(sdf, device=device), requires_grad=True)
        
        if (ncube_fg == self.target_ncube_fg):   # only final fine-grid can deform & use reg weight
            self.deform = torch.nn.Parameter(torch.zeros_like(self.marching_verts), requires_grad=True)
            self.per_cube_weights = torch.nn.Parameter(
                torch.ones((self.indices.shape[0], 21), dtype=torch.float32, device=device), requires_grad=True)
        else:
            self.deform, self.per_cube_weights = None, None
        
        self.max_scale = 2.
        self.scale_factor = torch.nn.Parameter(
            torch.arctanh(torch.full((1,), 1/self.max_scale, device=device)), requires_grad=True)

        torch.cuda.empty_cache()

    def create_from_mesh(self, training_args, coarse_mesh_path, model_path):
        # 1. load coarse mesh
        coarse_mesh = trimesh.load_mesh(coarse_mesh_path)
        
        # 2. init geometry(sdf, deform)
        self.target_ncube_fg = training_args.res_fg**3
        n_c2f_steps = len(training_args.c2f_steps)
        aabb_fg = torch.tensor(coarse_mesh.vertices.min(axis=0).tolist() +
                               coarse_mesh.vertices.max(axis=0).tolist(), device='cuda') * 1.1
        aabb_fg = aabb_fg.view(2, 3)
        print(f"Begin to initialize Geometry")
        self.init_geo(
            # init sdf (watertight mesh)
            coarse_mesh=coarse_mesh,
            # [foreground] differentiable geometry
            aabb_fg=aabb_fg,
            ncube_fg=self.get_cube_n(self.target_ncube_fg, n_c2f_steps, -1),
        )

        # 3. texture
        n_c = 3*((self.max_sh_degree + 1) ** 2)  # 3
        self.mlp_texture = MLPTexture3D(aabb_fg, channels=n_c, min_max=None)

        self.training_setup(training_args)

        # for debugging
        os.makedirs(model_path, exist_ok=True)
        self.export_mesh(os.path.join(model_path, "init_mesh.obj"))

        # for ablation study
        self.adaptive_cov = training_args.adaptive_cov
        print("Adaptive Covariance:", self.adaptive_cov)

    def compute_sdf_reg_loss(self, sdf, edges):
        sdf_f1x8x2 = sdf[edges.reshape(-1)].reshape(-1,2)
        mask = torch.sign(sdf_f1x8x2[...,0]) != torch.sign(sdf_f1x8x2[...,1])
        sdf_f1x8x2 = sdf_f1x8x2[mask]
        sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,0], (sdf_f1x8x2[...,1] > 0).float()) + \
                torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,1], (sdf_f1x8x2[...,0] > 0).float())
        return sdf_diff
    
    def getMesh(self):
        if self.deform is None:
            v_deformed = self.marching_verts
        else:
            v_deformed = self.marching_verts + self.max_displacement * torch.tanh(self.deform)
        if self.per_cube_weights is not None:
            w_1, w_2, w_3 = self.per_cube_weights[:,:12], self.per_cube_weights[:,12:20], self.per_cube_weights[:,20],
        else:
            w_1, w_2, w_3 = None, None, None
        verts, faces, reg_loss, surf_edges = self.marching_geo(v_deformed, self.sdf, self.indices, self.grid_res, 
                                                               w_1, w_2, w_3, training=True)

        flexi_reg_loss = reg_loss.mean()
        sdf_reg_loss = self.compute_sdf_reg_loss(self.sdf, surf_edges)

        return verts, faces, flexi_reg_loss, sdf_reg_loss

    def renew_gaussian(self, train_mesh):
        # -------------------------------------------------- #
        # 1. sdf/deform -> verts -> gaussians
        # 2. sample gaussian colors in 3D-texture
        # -------------------------------------------------- #
        device = 'cuda'
        with torch.set_grad_enabled(train_mesh):
            verts, faces, flexi_reg_loss, sdf_reg_loss = self.getMesh()

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
            L = M_2d * s_scalar.view(-1, 1, 1)  # [n_face, 2, 2] * [n_tri, 1, 1]

            cov3D_L = torch.zeros((n_face, 3, 3), device=device)
            cov3D_L[:, :2, :2] = L.view(n_face, 2, 2)
            cov3D_L[:, 2, 2] = self.spatial_lr_scale * 1e-6  # flat

        # gaussian colors
        features = self.mlp_texture.sample_noact(gs_xyz).view(N_gs, 3, (self.max_sh_degree + 1) ** 2)

        # gaussian opacity (triangle space)
        opacities = torch.full((N_gs, 1), 0.9999, device=device)

        # -------------------------------------------------- #
        # return dynamically generated gaussian information
        # -------------------------------------------------- #
        gs_info = {
            'xyz': gs_xyz,
            'opacity': opacities,
            'covariance': self.get_covariance_dyn(rot_t2w, cov3D_L),
            'cov3D_L': cov3D_L,
            'features': features,
            'active_sh_degree': self.active_sh_degree,
            'max_sh_degree': self.max_sh_degree,
            'flexi_reg_loss': flexi_reg_loss,
            'sdf_reg_loss': sdf_reg_loss,
            'verts': verts,
            'faces': faces,
            'fg_bg_nfaces': torch.tensor([faces.shape[0], 0], device='cuda'),
        }
        return gs_info

    @torch.no_grad()
    def export_mesh(self, path):
        verts, faces, _, _ = self.getMesh()
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                               faces=faces.detach().cpu().numpy())
        with open(path, 'w') as file:
            mesh.export(file_obj=file, file_type='obj')


    @property
    def get_scaling(self):
        raise NotImplementedError
    
    @property
    def get_rotation(self):
        raise NotImplementedError  # need to do transformation: mesh->world
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        raise NotImplementedError('should not call this function')
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        raise NotImplementedError('should not call this function')
        return self._opacity

    def get_covariance_dyn(self, rot_t2w, cov3D_L, scaling_modifier=1):
        assert scaling_modifier==1, "not supported"

        n_triangle = rot_t2w.shape[0]
        n_inner_gs = self.bc_coords.shape[0]

        # per triangle operations
        L = cov3D_L.view(n_triangle, 3, 3) * (torch.tanh(self.scale_factor) * self.max_scale)
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
        self.spatial_lr_scale = spatial_lr_scale  # also cam traj radius

    def training_setup(self, opt):
        l = [
            {'params': [self.sdf], 'lr': opt.sdf_lr, "name": "sdf"},
            # {'params': [self.deform], 'lr': opt.deform_lr, "name": "deform"},
            # {'params': [self.per_cube_weights], 'lr': opt.cube_weight_lr, "name": "per_cube_weights"},
            {'params': list(self.mlp_texture.parameters()), 'lr': opt.texture_lr, "name": "texture"},
            {'params': [self.scale_factor], 'lr': opt.scale_factor_lr, "name": "scale_factor"},
        ]
        if self.deform is not None:
            l += [
                {'params': [self.deform], 'lr': opt.deform_lr, "name": "deform"}, 
                {'params': [self.per_cube_weights], 'lr': opt.cube_weight_lr, "name": "per_cube_weights"},
            ]
        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.init_lr_collector = {item['name']:item['lr'] for item in l}
        self.scheduler_args = get_expon_lr_func(
            lr_init=1.0, lr_final=opt.final_lr_rate, max_steps=opt.reg_decay_iter)

    def update_learning_rate(self, iteration):
        lr_decay_rate = self.scheduler_args(iteration)
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["sdf"]:
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
