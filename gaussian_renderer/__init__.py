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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2D
from diff_surfel_rasterization import GaussianRasterizer as GaussianRasterizer2D

from scene.gaussian_model import GaussianModel

def render_gsinfo(viewpoint_camera, gs_info : dict, pipe, bg_color : torch.Tensor, image_only=False, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    xyz = gs_info['xyz']
    opacity = gs_info['opacity']
    scaling = gs_info['scaling'] if "scaling" in gs_info else None
    rotation = gs_info['rotation']  if "rotation" in gs_info else None
    features = gs_info['features']
    active_sh_degree = gs_info['active_sh_degree']

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gs_info['covariance']  # debug
    else:
        scales = scaling
        rotations = rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            raise NotImplementedError
            # shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            # dir_pp = (xyz - viewpoint_camera.camera_center.cuda().repeat(features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.sigmoid(sh2rgb)
            # # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": None,
            "alpha": None,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_gsinfo_surfel(viewpoint_camera, gs_info : dict, pipe, bg_color : torch.Tensor, image_only=False, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    xyz = gs_info['xyz']
    opacity = gs_info['opacity']
    scaling = gs_info['scaling'] if "scaling" in gs_info else None
    rotation = gs_info['rotation']  if "rotation" in gs_info else None
    features = gs_info['features']
    active_sh_degree = gs_info['active_sh_degree']

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings2D(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer2D(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gs_info['covariance']  # debug
    else:
        scales = scaling
        rotations = rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            raise NotImplementedError
            # shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            # dir_pp = (xyz - viewpoint_camera.camera_center.cuda().repeat(features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.sigmoid(sh2rgb)
            # # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "depth": None,
            "alpha": None,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
    }
    
    # # additional regularizations
    # render_alpha = allmap[1:2]

    # # get normal map
    # # transform normal from view space to world space
    # render_normal = allmap[2:5]
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # # get median depth map
    # render_depth_median = allmap[5:6]
    # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # # get expected depth map
    # render_depth_expected = allmap[0:1]
    # render_depth_expected = (render_depth_expected / render_alpha)
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # # get depth distortion map
    # render_dist = allmap[6:7]

    # # psedo surface attributes
    # # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    # # pipe.depth_ratio = 0  # TODO: make it adjustable
    # surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = surf_normal.permute(2,0,1)
    # # remember to multiply with accum_alpha since render_normal is unnormalized.
    # surf_normal = surf_normal * (render_alpha).detach()


    # rets.update({
    #         'rend_alpha': render_alpha,
    #         'rend_normal': render_normal,
    #         'rend_dist': render_dist,
    #         'surf_depth': surf_depth,
    #         'surf_normal': surf_normal,
    # })

    return rets


def render_raw(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rotation = pc.get_rotation
    features = pc.get_features
    active_sh_degree = pc.active_sh_degree

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        pass
    else:
        scales = scaling
        rotations = rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            raise NotImplementedError
            # shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            # dir_pp = (xyz - viewpoint_camera.camera_center.cuda().repeat(features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.sigmoid(sh2rgb)
            # # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "depth": rendered_depth,
            # "alpha": rendered_opacity,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
