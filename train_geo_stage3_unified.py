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

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_gsinfo
import sys
from scene import Scene
from scene.gs_geo_fine_unified import GaussianGeoModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import visualize_depth
from torchvision.utils import make_grid, save_image
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    ggeo = GaussianGeoModel(args.sh_degree)
    scene = Scene(args, ggeo)
    ggeo.create_from_geo_and_gs(args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc=f"{args.shape_name}-fine", dynamic_ncols=True)
    first_iter += 1

    for iteration in range(first_iter, args.iterations + 1):
        iter_start.record()

        if iteration in args.reset_opacity_steps:
            print('\n reset foreground opacity.')
            ggeo.reset_opacity(args)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        ggeo.update_learning_rate(iteration)
        gs_info = ggeo.renew_gaussian(viewpoint_cam=viewpoint_cam)

        # Render
        bg = torch.rand((3), device="cuda") if args.random_background else background

        render_pkg = render_gsinfo(viewpoint_cam, gs_info, args, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if render_pkg["depth"] is not None:
            depth = render_pkg["depth"] / render_pkg["alpha"].clamp_min(1e-5)
        
        if iteration==1 or iteration%args.save_image_interval==0:
            far = 2*scene.cameras_extent
            os.makedirs(f'{scene.model_path}/img', exist_ok=True)
            if render_pkg["depth"] is not None:
                grid = make_grid([image, visualize_depth(depth[0], far=far)], nrow=1)
            else:
                grid = image
            save_image(grid, f'{scene.model_path}/img/_latest_d.png')
            save_image(image, f'{scene.model_path}/img/{iteration:0>5}.png')
        if iteration%args.save_mesh_interval==0:
            os.makedirs(f'{scene.model_path}/mesh', exist_ok=True)
            ggeo.export_mesh(f'{scene.model_path}/mesh/{iteration:0>5}.obj')

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Reg Loss
        if (args.lambda_laplace_s3>0 or args.lambda_normal_consistent_s3>0) and gs_info['nfaces']>0:
            mesh = Meshes(verts=[gs_info['verts']], faces=[gs_info['faces']], textures=None)
            if args.lambda_laplace_s3 > 0:
                loss += args.lambda_laplace_s3 * mesh_laplacian_smoothing(mesh, method='uniform')
            if args.lambda_normal_consistent_s3 > 0:
                loss += args.lambda_normal_consistent_s3 * mesh_normal_consistency(mesh)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            progress_bar.set_postfix({
                "Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(1)
            if iteration == args.iterations:
                progress_bar.close()

            # Log and save
            def render_fn_warp(viewpoint, gaussians, *renderArgs):
                # gs_info = gaussians.renew_gaussian(viewpoint_cam=viewpoint)
                gs_info = renderArgs[2]
                return render_gsinfo(viewpoint, gs_info, renderArgs[0], renderArgs[1])
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            args.test_iterations, scene, render_fn_warp, (args, background, gs_info))
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < args.iterations:
                ggeo.optimizer.step()
                ggeo.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                gs_info = ggeo.renew_gaussian(viewpoint_cam=None)
                torch.save((ggeo.capture(gs_info), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                os.makedirs(f'{scene.model_path}/mesh', exist_ok=True)
                ggeo.export_mesh(f'{scene.model_path}/mesh/{iteration:0>5}.obj')
        del gs_info  # important to save memory

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    OmegaConf.save(config=args, f=os.path.join(args.model_path, "config.yaml"))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    args = renderArgs[0]
    # Report test and samples of training set
    if iteration in testing_iterations:
        last_test = (iteration == testing_iterations[-1])
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
                if last_test:
                    print("\n[ITER {}] Saving Testing Metrics".format(iteration))
                    with open(os.path.join(args.model_path, f"{config['name']}_metrics.json"), "w") as f:
                        json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item()}, f)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(1000, 50_000, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # list(range(1_000, 10_000, 3_000))
    parser.add_argument("--save_image_interval", type=int, default=100)    
    parser.add_argument("--save_mesh_interval", type=int, default=1000)

    args, _ = parser.parse_known_args(sys.argv[1:])
    vanilla_gs_config = OmegaConf.create(vars(args))
    config = OmegaConf.load(args.config)
    if 'base_config' in config:
        base_path = os.path.join(os.path.dirname(args.config), config['base_config'])
        if os.path.exists(base_path):
            config = OmegaConf.merge(OmegaConf.load(base_path), config)
    cli = OmegaConf.from_cli()
    args = OmegaConf.merge(vanilla_gs_config, config, cli)

    # added
    args.compute_cov3D_python = False
    args.convert_SHs_python = False
    args.sh_degree = 3
    args.model_path = args.s3_model_path
    args.iterations = args.s3_iterations
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(True)
    training(args)

    # All done
    print("\nTraining complete.")
