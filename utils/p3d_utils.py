import torch
import numpy as np
import math
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Textures,
    FoVPerspectiveCameras as P3DCameras,
    MeshRasterizer,
    RasterizationSettings,
)
# from pytorch3d.renderer.opengl.rasterizer_opengl import (
#     MeshRasterizerOpenGL,
# )

def render_depth(verts, faces, p3d_cam, height, width):
    mesh = Meshes(verts=[verts], faces=[faces])
    rasterizer = MeshRasterizer(
        cameras=p3d_cam,
        raster_settings=RasterizationSettings(
            image_size=[height, width],
            blur_radius=0.0,
            faces_per_pixel=1,  # top-1
            cull_backfaces=False,
            max_faces_per_bin=10000,
            bin_size=None,  # 0 for naive rast, but slow
        ),
    )
    # rasterizer = MeshRasterizerOpenGL(
    #     cameras=p3d_cam,
    #     raster_settings=RasterizationSettings(
    #         image_size=[height, width],
    #         blur_radius=0.0,
    #         faces_per_pixel=1,  # top-1
    #         cull_backfaces=False,
    #         bin_size=0,  # 0 for naive rast, but slow
    #     ),
    # )
    fragments = rasterizer(meshes_world=mesh)
    return fragments.zbuf[0, :, :, 0]


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """
    Convert mesh normals to RGB color representation.

    Args:
        normals (torch.Tensor): Mesh normals.

    Returns:
        torch.Tensor: RGB colors based on the normals.
    """
    return torch.abs(normals * 0.5 + 0.5)


def get_normals_as_textures(mesh: Meshes) -> Meshes:
    """
    Create textures from mesh normals.

    Args:
        mesh (Meshes): Input mesh.

    Returns:
        Meshes: New mesh with normals as textures.
    """
    normals = mesh.verts_normals_packed()
    textures = Textures(verts_rgb=normals_to_rgb(normals).unsqueeze(0))
    return Meshes(mesh.verts_packed().unsqueeze(0), mesh.faces_packed().unsqueeze(0), textures)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras