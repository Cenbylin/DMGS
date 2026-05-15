import os
import torch
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from omegaconf import OmegaConf

import open3d as o3d
import numpy as np
import point_cloud_utils as pcu
import trimesh

from train_geo_stage1 import training as s1_training
from train_geo_stage2_unified import training as s2_training
from train_geo_stage2a_unified import training as s2a_training
from train_geo_stage3_unified import training as s3_training

def s1_post(args):
    (model_params, _) = torch.load(args.coarse_gs_path, weights_only=False)
    (active_sh_degree, xyz, _features_dc, _features_rest, _scaling, _rotation, opacity, 
     max_radii2D, xyz_gradient_accum, denom,opt_dict, spatial_lr_scale) = model_params
    device = xyz.device
    # filter xyz
    xyz = xyz[torch.sigmoid(opacity[:, 0])>0.1]
    if args.aabb_fg is not None:
        xyz_min, xyz_max = torch.tensor(args.aabb_fg, device=device).view(2, 3)
        inbox_mask = (xyz_min<xyz).all(dim=1) & (xyz<xyz_max).all(dim=1)
        xyz = xyz[inbox_mask]
    # alpha shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().double().cpu().numpy())
    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, args.alpha_shape)
    # watertight
    vw, fw = pcu.make_mesh_watertight(np.asarray(o3d_mesh.vertices), 
                                      np.asarray(o3d_mesh.triangles), args.watertight_res)
    # export
    mesh = trimesh.Trimesh(vw, fw)
    with open(args.coarse_mesh_path, 'w') as file:
        mesh.export(file_obj=file, file_type='obj')

def save_time_count(args, time_count):
    path = os.path.join(args.model_path, 'time_count.txt')
    with open(path, 'w') as f:
        f.write(str(time_count))

def pipeline(args):
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # original 3DGS
    if '1' in args.stages:
        iter_start.record()
        print(f"================ Stage 1 ================")
        args.model_path = args.s1_model_path
        args.iterations = args.s1_iterations
        args.checkpoint_iterations = [args.iterations]
        s1_training(args, args, args, [], [], 
                    args.checkpoint_iterations, None, -1)
        s1_post(args)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)/60_000
        print(f"Stage 1 elapsed time: {elapsed_time:.2f} mins")
        save_time_count(args, elapsed_time)
        torch.cuda.empty_cache()

    # mesh & appearance learning
    if '2' in args.stages:
        iter_start.record()
        print(f"================ Stage 2 ================")
        args.model_path = args.s2_model_path
        args.iterations = args.s2_iterations
        args.checkpoint_iterations = [args.iterations]
        args.test_iterations = [args.iterations]
        s2_training(args)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)/60_000
        print(f"Stage 2 elapsed time: {elapsed_time:.2f} mins")
        save_time_count(args, elapsed_time)
        torch.cuda.empty_cache()

    # adaption to new scene
    if '2a' in args.stages:
        iter_start.record()
        print(f"================ Stage 2a ================")
        args.source_path = args.adapt_path
        args.model_path = args.s2a_model_path
        args.iterations = args.s2_iterations
        args.checkpoint_iterations = [args.iterations]
        args.test_iterations = [args.iterations]
        s2a_training(args)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)/60_000
        print(f"Stage 2 elapsed time: {elapsed_time:.2f} mins")
        save_time_count(args, elapsed_time)
        torch.cuda.empty_cache()
    
    # refiment for better appearance
    if '3' in args.stages:
        iter_start.record()
        print(f"================ Stage 3 ================")
        args.data_device = 'cuda'
        args.model_path = args.s3_model_path
        args.iterations = args.s3_iterations
        args.checkpoint_iterations = [args.iterations]
        args.test_iterations = [args.iterations]
        s3_training(args)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)/60_000
        print(f"Stage 3 elapsed time: {elapsed_time:.2f} mins")
        save_time_count(args, elapsed_time)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default=None, help="Yaml Config file")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_image_interval", type=int, default=1000)    
    parser.add_argument("--save_mesh_interval", type=int, default=999999)
    parser.add_argument("--stages", nargs="+", type=str, default=['1', '2', '3'],)

    # merge all arguments
    args, _ = parser.parse_known_args(sys.argv[1:])
    vanilla_gs_config = OmegaConf.create(vars(args))
    config = OmegaConf.load(args.config)
    if 'base_config' in config:
        base_path = os.path.join(os.path.dirname(args.config), config['base_config'])
        if os.path.exists(base_path):
            config = OmegaConf.merge(OmegaConf.load(base_path), config)
    cli = OmegaConf.from_cli()
    args = OmegaConf.merge(vanilla_gs_config, config, cli)

    # Initialize system state (RNG)
    safe_state(False)
    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(True)

    pipeline(args)

    print("\nTraining complete.")