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
from utils.graphics_utils import BasicPointCloud
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from utils.effi_utils import count_time
from scene.core import bind_2dgs_to_face

import trimesh
import open3d as o3d

import geo.mesh_utils as mesh_utils
from geo.texture import SHField
import torch.nn.functional as F

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class GaussianGeoModel(nn.Module):
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        super().__init__()
        self.active_sh_degree = sh_degree
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
        )
    
    def restore(self, model_args, training_args):
        raise NotImplementedError  # paired with capture()

    @torch.no_grad()
    def load_for_eval(self, model_params):
        (_xyz, _features, _opacity, _scaling, _rotation, _, _, npts_bg, npts_fg,
         _, active_sh_degree, _, _, _) = model_params

        return {
            'xyz': _xyz,
            'opacity':  _opacity,
            'scaling':  _scaling,
            'rotation': _rotation,
            'features': _features,
            'active_sh_degree': active_sh_degree,
            'npts_bg': npts_bg,
            'npts_fg': npts_fg,
        }  # gs_info


    @torch.no_grad()
    def create_from_geo_and_gs(self, args):
        # 1. mesh & gaussians info
        s2_chkpnt_path = f"{args.s2_model_path}/chkpnt{args.s2_iterations}.pth"
        (model_params, _) = torch.load(s2_chkpnt_path, weights_only=False)
        (_xyz, _features, _opacity, _scaling, _rotation, verts, faces, npts_bg, npts_fg,
         _, self.active_sh_degree, self.spatial_lr_scale, bc_coords, rad_base, AABB, mlp_texture_state) = model_params
        
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                               faces=faces.detach().cpu().numpy())
        with open(f'{args.model_path}/mesh_init.obj', 'w') as file:
            mesh.export(file_obj=file, file_type='obj')

        # -------------------------------------------------- #
        # pre-process the mesh, bake texture to new Gaussians
        # -------------------------------------------------- #
        # options
        need_subdivision = (args.simplify_nface>len(faces))
        need_simplify_mesh = args.simplify_nface>0  # if have target nface
        need_new_bind_rule = args.s3_gs_per_face != bc_coords.shape[0]
        
        if need_subdivision:
            o3d_mesh_fg = o3d.geometry.TriangleMesh()
            o3d_mesh_fg.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
            o3d_mesh_fg.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
            o3d_mesh_fg = o3d_mesh_fg.remove_unreferenced_vertices()
            o3d_mesh_fg = o3d_mesh_fg.remove_degenerate_triangles()
            o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_triangles()
            o3d_mesh_fg = o3d_mesh_fg.remove_duplicated_vertices()
            o3d_mesh_fg = o3d_mesh_fg.remove_non_manifold_edges()
            o3d_mesh_fg = o3d_mesh_fg.subdivide_loop(1)
            o3d_mesh = o3d_mesh_fg
            mesh_new = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                       faces=np.asarray(o3d_mesh.triangles))
            new_n_face = len(mesh_new.faces)
            print(f'subdividing from {len(faces)} to {new_n_face} faces')
            with open(f'{args.model_path}/mesh_subdivided.obj', 'w') as file:
                mesh_new.export(file_obj=file, file_type='obj')
            verts = torch.from_numpy(mesh_new.vertices).to(verts)
            faces = torch.from_numpy(mesh_new.faces).to(faces)
        
        if need_simplify_mesh:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(args.simplify_nface)
            o3d_mesh = o3d_mesh.remove_degenerate_triangles()
            o3d_mesh = o3d_mesh.remove_duplicated_triangles()
            o3d_mesh = o3d_mesh.remove_duplicated_vertices()
            o3d_mesh = o3d_mesh.remove_non_manifold_edges()
            mesh_new = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), 
                                       faces=np.asarray(o3d_mesh.triangles))
            
            print(f'simplified from {len(faces)} to {len(mesh_new.faces)} faces')
            with open(f'{args.model_path}/mesh_simplified.obj', 'w') as file:
                mesh_new.export(file_obj=file, file_type='obj')
            verts = torch.from_numpy(mesh_new.vertices).to(verts)
            faces = torch.from_numpy(mesh_new.faces).to(faces)
            fg_bg_nfaces = torch.tensor([faces.shape[0], 0], device='cuda')
            
        if need_new_bind_rule:
            # bake texture to new Gaussians
            bc_coords, rad_base = mesh_utils.generate_barycentric_v2(args.s3_gs_per_face, 'cuda')

        print(f'bake texture to new 2D Gaussians')
        mlp_texture = SHField(AABB, sh_degree=self.active_sh_degree, min_max=None, gradient_scaling=1.0)
        mlp_texture.load_state_dict(mlp_texture_state)
        self.bc_coords = bc_coords
        self.rad_base = rad_base
        with torch.no_grad():
            fg_xyz, fg_scale_2D, fg_rot_2D, fg_features, fg_opacities_acted = \
                bind_2dgs_to_face(verts, faces, bc_coords, rad_base, self.spatial_lr_scale, mlp_texture)

        # -------------------------------------------------- #
        # construct parameters and config optimizer
        # -------------------------------------------------- #
        # ------ preprocess ------ #
        bg_features_dc, bg_features_rest = _features[:npts_bg, 0:1, :], _features[:npts_bg, 1:, :]
        fg_features_dc, fg_features_rest = fg_features[:, 0:1, :], fg_features[:, 1:, :]

        # ------ composed ------ #
        self._features_dc = nn.Parameter(
            torch.cat([bg_features_dc, fg_features_dc], dim=0).detach().clone().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.cat([bg_features_rest, fg_features_rest], dim=0).detach().clone().requires_grad_(True))
        self._opacity = nn.Parameter(inverse_sigmoid(
            torch.cat([_opacity[:npts_bg], fg_opacities_acted], dim=0)
        ).detach().clone().requires_grad_(True))
        
        # ------ runtime composing ------ #
        self.bg_xyz = nn.Parameter(_xyz[:npts_bg].detach().clone().requires_grad_(True))
        self.bg_scaling = nn.Parameter(self.scaling_inverse_activation(_scaling[:npts_bg]).detach().clone().requires_grad_(True))
        self.bg_rotation = nn.Parameter(_rotation[:npts_bg].detach().clone().requires_grad_(True))
        self.fg_scaling = nn.Parameter(self.scaling_inverse_activation(fg_scale_2D).detach().clone().requires_grad_(True))
        self.fg_rotation = nn.Parameter(fg_rot_2D.detach().clone().requires_grad_(True))

        # ------ others ------ #
        self.verts = nn.Parameter(verts.detach().clone().requires_grad_(True))
        self.faces = faces

        self.training_setup(args)
    
    def renew_gaussian(self, viewpoint_cam=None):
        # ------------------------ #
        # verts -> gaussians
        # ------------------------ #
        verts, faces = self.verts, self.faces

        # mesh -> foreground 3d gs
        face_vert = verts[faces]
        v1v2 = face_vert[:, 1, :] - face_vert[:, 0, :]
        v1v2_len = torch.norm(v1v2, dim=-1, keepdim=True).clamp_min(1e-12)
        face_x = torch.div(v1v2, v1v2_len)
        face_normals = mesh_utils.face_normals(verts, faces, unit=True)
        face_y = torch.nn.functional.normalize(torch.cross(face_normals, face_x, dim=-1))
        rot_t2w = torch.stack([face_x, face_y, face_normals], dim=2)
        fg_xyz = torch.matmul(self.bc_coords.view(1, -1, 3), face_vert).view(-1, 3)
        thin_z_scale = self.scaling_inverse_activation(torch.tensor(self.spatial_lr_scale*1e-6))
        fg_scaling = torch.cat([
            self.fg_scaling, torch.full_like(self.fg_scaling[:, :1], thin_z_scale)
        ], dim=1)
        # 2d rot to quat
        a, b = self.fg_rotation[..., 0], self.fg_rotation[..., 1]
        half_theta = torch.atan2(b, a) / 2.0
        fg_rotation = torch.stack([
            torch.cos(half_theta), torch.zeros_like(half_theta), 
            torch.zeros_like(half_theta), torch.sin(half_theta)
        ], dim=-1)  # already normed quat
        nfaces = rot_t2w.size(0)
        fg_rotation = quaternion_multiply(
            matrix_to_quaternion(rot_t2w).view(nfaces, 1, 4), 
            fg_rotation.view(nfaces, -1, 4)
        ).view(-1, 4)
        
        # composing
        gs_info = {
            'xyz': torch.cat([self.bg_xyz, fg_xyz], dim=0),
            'opacity':  self.opacity_activation(self._opacity),
            'scaling':  self.scaling_activation(torch.cat([self.bg_scaling, fg_scaling], dim=0)),
            'rotation': self.rotation_activation(torch.cat([self.bg_rotation, fg_rotation], dim=0)),
            'features': torch.cat([self._features_dc, self._features_rest], dim=1),
            'active_sh_degree': self.active_sh_degree,
            'npts_bg': self.bg_xyz.size(0),
            'npts_fg': fg_xyz.size(0),
            'verts': verts,
            'faces': faces,
            'nfaces': faces.size(0),
        }
        return gs_info


    @torch.no_grad()
    def export_mesh(self, path):
        verts, faces = self.verts, self.faces
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), 
                               faces=faces.detach().cpu().numpy())
        with open(path, 'w') as file:
            mesh.export(file_obj=file, file_type='obj')
        torch.cuda.empty_cache()

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # the only used term
        self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, opt):
        l = [
            # composed
            {'params': [self._features_dc], 'lr': opt.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opt.opacity_lr, "name": "opacity"},
            # foreground
            {'params': [self.fg_scaling], 'lr': opt.scaling_lr, "name": "fg_scaling"},
            {'params': [self.fg_rotation], 'lr': opt.rotation_lr, "name": "fg_rotation"},
            # mesh
            {'params': [self.verts], 'lr': opt.vert_lr, "name": "verts"},
        ]
        if len(self.bg_xyz) > 0:
            l += [
                # background
                {'params': [self.bg_xyz], 'lr': opt.position_lr_final * self.spatial_lr_scale, "name": "bg_xyz"},
                {'params': [self.bg_scaling], 'lr': opt.scaling_lr, "name": "bg_scaling"},
                {'params': [self.bg_rotation], 'lr': opt.rotation_lr, "name": "bg_rotation"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if opt.vert_lr_final is None:
            opt.vert_lr_final = opt.vert_lr
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.vert_lr,
                                                    lr_final=opt.vert_lr_final,
                                                    max_steps=opt.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "verts":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

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

    def reset_opacity(self, args):
        # only reset fg
        npts_bg = self.bg_xyz.size(0)
        fg_opacities_new = inverse_sigmoid(torch.full_like(self._opacity[npts_bg:], args.init_opacity))
        opacities_new = torch.cat([self._opacity[:npts_bg], fg_opacities_new], dim=0).detach().clone()

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
