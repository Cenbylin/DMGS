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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric
from scene.core import bind_gs_to_face_knn
from utils.effi_utils import count_time

import trimesh
import point_cloud_utils as pcu
from geo.flexicubes_no_adjacent import FlexiCubes

import geo.mesh_utils as mesh_utils
from geo.texture import SHField
import torch.nn.functional as F

@torch.no_grad()
@torch.jit.script
def in_frustum(proj_matrix: torch.Tensor, verts: torch.Tensor, cube_len: float):
    # following in_frustum() in cuda_rasterize
    # method-1 use projection matrix.
    proj_matrix_3x3 = proj_matrix[:3, :]
    # trick: we fuse the `cube center = cube_len/2. + verts` into the proj_matrix, to save time&memory
    # so, below = (vert + cube_len/2.) @ proj_matrix_3x3 + proj_matrix_1x3
    proj_matrix_1x3 = proj_matrix[3:, :] + (proj_matrix_3x3*(cube_len/2.)).sum(dim=0, keepdim=True)
    p_proj = torch.matmul(verts, proj_matrix_3x3)
    p_proj += proj_matrix_1x3  #  # [N, 4] @ [4, 4], use right-mul
    w = (p_proj[:, 3:] + 1e-6)
    p_proj = p_proj[:, :3] / w

    mask = (p_proj.abs()<1.05).all(dim=-1)  # within screen space
    mask &= (w.squeeze()>0)  # in front of camera

    return mask

class GaussianGeoModel(nn.Module):
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, gs_per_face, c2f_rate, scale_factor=1.0):
        super().__init__()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        # for logging
        self.max_frame_nface = 0

        # added
        self.sdf = None
        self.deform = None
        self.last_sdf = None  # for tracking topology changes
        self.bc_coords, self.rad_base = mesh_utils.generate_barycentric_v2(gs_per_face, 'cuda')
        self.rad_base *= scale_factor
        self.c2f_rate = c2f_rate

        self.setup_functions()

    def capture(self, gs_info):
        return (
            # last matched result
            gs_info['xyz'],
            gs_info['features'],
            gs_info['opacity'],  # activated
            gs_info['scaling'],  # activated
            gs_info['rotation'],  # activated
            gs_info['verts'], 
            gs_info['faces'],
            gs_info['npts_bg'],
            gs_info['npts_fg'],

            # model states
            self.optimizer.state_dict(),
            self.active_sh_degree,
            self.spatial_lr_scale,
            self.bc_coords,
            self.rad_base,
            self.mlp_texture.AABB,
            self.mlp_texture.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        raise NotImplementedError  # paired with capture()


    def init_geo(self, coarse_mesh, aabb_fg, ncube_fg):
        device = "cuda"
        # extrator
        self.marching_geo    = FlexiCubes()
        # aabb
        self.aabb_fg = torch.tensor(aabb_fg, device='cuda').view(2, 3)
        # coarse mesh info
        vw, fw = coarse_mesh.vertices.astype(np.float32), coarse_mesh.faces
        
        #####################################
        # foreground
        #####################################
        # 1. grid
        (self.grid_res, self.marching_verts, self.cube_edge_fg,
         self.max_displacement) = self.setup_grid(self.aabb_fg, ncube_fg)
        self.cube_ind_cpu = self.marching_geo.construct_cubes(
                torch.ones_like(self.marching_verts[:, 0], dtype=torch.bool), self.grid_res).to('cpu')
        # 2. Learnable parts
        sdf, fid, bc = pcu.signed_distance_to_mesh(self.marching_verts.cpu().numpy(), vw, fw)
        self.sdf = torch.nn.Parameter(torch.tensor(sdf, device=device), requires_grad=True)
        self.deform_fg = torch.nn.Parameter(torch.zeros_like(self.marching_verts), requires_grad=True) \
                if (ncube_fg == self.target_ncube_fg) else None  # only final fine-grid can deform
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

        self.training_setup(opt)

    def setup_grid(self, aabb, ncube):
        # 1. xyz range and resolutions
        aabb_edge = (aabb[1] - aabb[0])
        cube_edge = float((aabb_edge.prod() / (ncube)) ** (1/3))
        res_fg_list = (aabb_edge / cube_edge).int().tolist()
        print("Foreground grid:", res_fg_list)

        # 2. init flexicube (differentiable iso-surface extrator)
        grid_res = res_fg_list
        marching_verts = self.marching_geo.construct_voxel_grid_vert_only(
            grid_res, aabb[0], aabb[1])
        max_displacement = (cube_edge / 2.)

        return grid_res, marching_verts, cube_edge, max_displacement

    def create_from_mesh_and_gs(self, args):
        # 1. load coarse mesh
        coarse_mesh = trimesh.load_mesh(args.coarse_mesh_path)
        
        # 2. init geometry (sdf, deform)
        self.enable_bg = (args.aabb_fg is not None)
        if not self.enable_bg:  # auto infer aabb for foreground-only scene.
            args.aabb_fg = [(coarse_mesh.vertices.min(axis=0) * 1.1).tolist(), 
                            (coarse_mesh.vertices.max(axis=0) * 1.1).tolist()]
        n_c2f_steps = len(args.c2f_steps)
        self.target_ncube_fg = args.res_fg**3
        print(f"Begin to initialize Geometry.")
        self.init_geo(
            # init sdf (watertight mesh)
            coarse_mesh=coarse_mesh, aabb_fg=args.aabb_fg,
            # [foreground] differentiable geometry
            ncube_fg=self.get_cube_n(self.target_ncube_fg, n_c2f_steps, -1),
        )

        # 3. texture
        AABB = torch.tensor(args.aabb_fg, device='cuda')
        assert AABB.shape == (2, 3)
        # self.mlp_texture = SHField(AABB, sh_degree=self.max_sh_degree, min_max=None, gradient_scaling=1.0)

        # 4. background gaussians
        (model_params, _) = torch.load(args.coarse_gs_path)
        (active_sh_degree, _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, 
         max_radii2D, xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params
        assert self.active_sh_degree == active_sh_degree
        if self.enable_bg:
            bg_mask = (_xyz < AABB[:1, :]).any(dim=1) | (_xyz > AABB[1:, :]).any(dim=1)
        else:
            bg_mask = torch.zeros_like(_xyz[:, 0], dtype=bool)
        self._xyz = nn.Parameter(_xyz[bg_mask].detach().clone().requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc[bg_mask].detach().clone().requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest[bg_mask].detach().clone().requires_grad_(True))
        self._scaling = nn.Parameter(_scaling[bg_mask].detach().clone().requires_grad_(True))
        self._rotation = nn.Parameter(_rotation[bg_mask].detach().clone().requires_grad_(True))
        self._opacity =nn.Parameter(_opacity[bg_mask].detach().clone().requires_grad_(True))

        # 5. additional
        # self.scale_factor = torch.nn.Parameter(torch.full((1,), 1., device='cuda'), requires_grad=True)
        self.training_setup(args)

        # 🍉 for debugging
        os.makedirs(args.model_path, exist_ok=True)
        self.export_mesh(os.path.join(args.model_path, "init_mesh.obj"))

        # 🍉 for ablation study
        self.adaptive_cov = args.adaptive_cov
        print("Adaptive Covariance:", self.adaptive_cov)

    def getMesh_in_frustum(self, viewpoint_cam, training):
        # make all out_furstum sdf be 1 (>0). they will be filtered during marching.
        vert_mask = in_frustum(viewpoint_cam.full_proj_transform.cuda(), 
                               self.marching_verts, self.cube_edge_fg)
        # vert_mask = torch.ones_like(self.marching_verts[:, 0], dtype=torch.bool)
        indices = self.marching_geo.construct_cubes(vert_mask, self.grid_res)
        if self.deform is None:
            v_deformed = self.marching_verts
        else:
            v_deformed = self.marching_verts + self.max_displacement * torch.tanh(self.deform)
        w_1, w_2, w_3 = None, None, None
        if hasattr(self, 'per_cube_weights') and self.per_cube_weights is not None:
            w_1, w_2, w_3 = self.per_cube_weights[:,:12], self.per_cube_weights[:,12:20], self.per_cube_weights[:,20],
            
        verts, faces, reg_loss, surf_edges = self.marching_geo(v_deformed, self.sdf, indices, self.grid_res, 
                                                               w_1, w_2, w_3, training=training)
        if surf_edges is not None:
            flexi_reg_loss = reg_loss.mean()
            sdf_reg_loss = self.compute_sdf_reg_loss(self.sdf, surf_edges)
        else:
            flexi_reg_loss, sdf_reg_loss = 0., 0.
        
        return verts, faces, flexi_reg_loss, sdf_reg_loss

    def getMesh_full(self):
        if self.deform is None:
            v_deformed = self.marching_verts
        else:
            v_deformed = self.marching_verts + self.max_displacement * torch.tanh(self.deform)
        verts, faces, reg_loss, surf_edges = self.marching_geo(
            v_deformed, self.sdf, self.cube_ind_cpu.cuda(), self.grid_res, training=False, de_ambiguity=False)
        
        if surf_edges is not None:
            flexi_reg_loss = reg_loss.mean()
            sdf_reg_loss = self.compute_sdf_reg_loss(self.sdf, surf_edges)
        else:
            flexi_reg_loss, sdf_reg_loss = 0., 0.

        return verts, faces, flexi_reg_loss, sdf_reg_loss
        
    def renew_gaussian(self, train_mesh, gs_info_prev, viewpoint_cam=None):
        # -------------------------------------------------- #
        # 1. sdf/deform -> verts -> gaussians
        # 2. sample gaussian colors in 3D-texture
        # -------------------------------------------------- #
        if viewpoint_cam is not None:
            # verts, faces, flexi_reg_loss, sdf_reg_loss = \
            #     self.getMesh_in_frustum(viewpoint_cam, train_mesh)
            # self.max_frame_nface = max(self.max_frame_nface, int(faces.shape[0]))
            raise NotImplementedError
        else:
            verts, faces, flexi_reg_loss, sdf_reg_loss = self.getMesh_full()  # no_grad
            self.max_frame_nface = max(self.max_frame_nface, int(faces.shape[0]))
        
        # 🍉 core
        if gs_info_prev is None:
            xyz_prev, feat_prev = None, None
        else:
            xyz_prev = gs_info_prev['xyz'][gs_info_prev['npts_bg']:]
            feat_prev = gs_info_prev['features_param']
        gs_xyz, _scaling_acted, _rotation, features, opacities_acted = \
            bind_gs_to_face_knn(verts, faces, self.bc_coords, self.rad_base, self.spatial_lr_scale,
                                xyz_prev, feat_prev, adaptive_cov=self.adaptive_cov)
        # let appearance learnable
        features = nn.Parameter(features.requires_grad_(True))
        feat_optimizer = torch.optim.Adam([features], lr=0.0025)
        
        # -------------------------------------------------- #
        # return dynamically generated gaussian information
        # -------------------------------------------------- #
        features_bg = torch.cat((self._features_dc, self._features_rest), dim=1)

        gs_info = {
            'xyz': torch.cat([self._xyz.detach(), gs_xyz], dim=0),
            'opacity':  torch.cat([self.opacity_activation(self._opacity).detach(), opacities_acted], dim=0),
            'scaling':  torch.cat([self.scaling_activation(self._scaling).detach(), _scaling_acted], dim=0),
            'rotation': self.rotation_activation(torch.cat([self._rotation.detach(), _rotation], dim=0)),
            'features_param': features,
            'features': torch.cat([features_bg.detach(), features], dim=0),
            'feat_optimizer': feat_optimizer,
            'active_sh_degree': self.active_sh_degree,
            'max_sh_degree': self.max_sh_degree,
            'flexi_reg_loss': flexi_reg_loss,
            'sdf_reg_loss': sdf_reg_loss,
            'verts': verts,
            'faces': faces,
            "nfaces": faces.size(0),
            "npts_bg": self._xyz.size(0),
            "npts_fg": gs_xyz.size(0),
        }
        return gs_info

    # debug
    def get_covariance_dyn(self, rot_t2w, cov3D_L):
        n_triangle = rot_t2w.shape[0]
        n_inner_gs = self.bc_coords.shape[0]

        L = cov3D_L.view(n_triangle, 3, 3)  # * sf_per_face.view(n_triangle, 1, 1)
        R_t2w = rot_t2w.view(n_triangle, 3, 3)  # world space R
        L = torch.matmul(R_t2w, L).view(-1, 3, 3)  # [n_triangle,3,3]
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)

        # shared by all inner gs
        symm = symm.view(n_triangle, 1, 6).expand(-1, n_inner_gs, -1).reshape(-1, 6)
        return symm

    def compute_sdf_reg_loss(self, sdf, edges):
        sdf_f1x8x2 = sdf[edges.reshape(-1)].reshape(-1,2)
        mask = torch.sign(sdf_f1x8x2[...,0]) != torch.sign(sdf_f1x8x2[...,1])
        sdf_f1x8x2 = sdf_f1x8x2[mask]
        sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,0], (sdf_f1x8x2[...,1] > 0).float()) + \
                torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x8x2[...,1], (sdf_f1x8x2[...,0] > 0).float())
        return sdf_diff

    @torch.no_grad()
    def export_mesh(self, path):
        verts, faces, _, _ = self.getMesh_full()
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                               faces=faces.detach().cpu().numpy())
        with open(path, 'w') as file:
            mesh.export(file_obj=file, file_type='obj')
        torch.cuda.empty_cache()
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # the only used term
        self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, opt):
        sdf_lr = opt.sdf_lr[0] if isinstance(opt.sdf_lr, list) else opt.sdf_lr

        l = [
            # background - none
            # foreground
            # {'params': list(self.mlp_texture.parameters()), 'lr': opt.texture_lr, "name": "texture"},
            {'params': [self.sdf], 'lr': sdf_lr, "name": "sdf"},
            # {'params': [self.scale_factor], 'lr': opt.scale_factor_lr, "name": "scale_factor"},
        ]
        if self.deform is not None:  # due to 'coarse to fine'
            l += [{'params': [self.deform], 'lr': opt.deform_lr, "name": "deform"}]
        
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
