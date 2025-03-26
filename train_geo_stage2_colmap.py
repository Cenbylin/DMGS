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
import torch
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_dyn, network_gui
import sys
from scene import Scene
from scene.gaussian_geo_model_mlp_flex_colmap import GaussianGeoModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParamsGeo
from torchvision.transforms import ToPILImage

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, warm_up_iter,
             testing_iterations, saving_iterations, checkpoint_iterations, 
             coarse_mesh_path, aabb_fg_bg, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    ggeo = GaussianGeoModel(dataset.sh_degree, opt.gs_per_face, opt.c2f_rate)
    scene = Scene(dataset, ggeo)

    assert coarse_mesh_path, "stage-2 need a watertight mesh."

    ggeo.create_from_mesh(opt, coarse_mesh_path, aabb_fg_bg, dataset.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=120)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, ggeo, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # increase the levels of SH to a maximum degree
        if iteration in [1,2,3]:
            ggeo.oneupSHdegree()

        if iteration < (warm_up_iter+1):
            ggeo.warm_up = True
            ggeo.train_fg = False
            ggeo.train_bg = True
        else:
            ggeo.warm_up = False
            ggeo.train_fg = True
            ggeo.train_bg = True

        if (iteration-1) in opt.c2f_steps:
            print("\nTriggered coarse to fine.")
            ggeo.coarse_to_fine(len(opt.c2f_steps), opt.c2f_steps.index(iteration-1), opt)
        
        ggeo.update_learning_rate(iteration)
        gs_info = ggeo.renew_gaussian(train_mesh=True, viewpoint_cam=viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_dyn(viewpoint_cam, gs_info, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if iteration==1 or iteration%100==0 or iteration==(warm_up_iter+1):
            img = ToPILImage()(image)
            os.makedirs(f'{scene.model_path}/img', exist_ok=True)
            img.save(f'{scene.model_path}/img/{iteration:0>5}.png')
            img.save(f'{scene.model_path}/img/latest.png')
        if iteration%1000==0:
            os.makedirs(f'{scene.model_path}/mesh', exist_ok=True)
            ggeo.export_mesh(f'{scene.model_path}/mesh/{iteration:0>5}.obj')

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ----------------------------- #
        # reg loss weighr decay
        # ----------------------------- #
        if opt.reg_decay_iter>0:
            decay_rate = min(1.0, iteration/opt.reg_decay_iter)
            lambda_sdf = opt.lambda_sdf[0]-(opt.lambda_sdf[0]-opt.lambda_sdf[1])*decay_rate
            lambda_flex = opt.lambda_flex[0]-(opt.lambda_flex[0]-opt.lambda_flex[1])*decay_rate
        else:
            lambda_sdf = opt.lambda_sdf
            lambda_flex = opt.lambda_flex
        
        # if iteration==(opt.reg_decay_iter+1):
        #     print("switch to next optimization stage.")
        #     ggeo.next_stage_setup(opt)

        loss += (lambda_sdf * gs_info['sdf_reg_loss'])
        loss += (lambda_flex * gs_info['flexi_reg_loss'])

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            nface_k = (gs_info['fg_bg_nfaces']/1000.).tolist()

            progress_bar.set_postfix({
                "Loss": f"{ema_loss_for_log:.{7}f}",
                "nface": f"({nface_k[0]:>3.0f}k+{nface_k[1]:>3.0f}k, Mx{ggeo.max_frame_nface//1000}k)",
                })
            progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            def render_fn_warp(viewpoint, gaussians, *renderArgs):
                gs_info = gaussians.renew_gaussian(train_mesh=False, viewpoint_cam=viewpoint)
                return render_dyn(viewpoint, gs_info, *renderArgs)
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render_fn_warp, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                ggeo.optimizer.step()
                ggeo.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                gs_info = ggeo.renew_gaussian(train_mesh=False)
                torch.save((ggeo.capture(gs_info), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        del gs_info  # important to save memory

def prepare_output_and_logger(args, opt, pipe):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))
    with open(os.path.join(args.model_path, "pipe_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))

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
        tb_writer.add_scalar('max_frame_nface', scene.gaussians.max_frame_nface, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
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

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParamsGeo(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=True)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(500, 20_000, 500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # list(range(1_000, 10_000, 3_000))
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--coarse_mesh_path", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    if args.config is not None:
        data = json.load(open(args.config, 'r'))
        for key in data:
            args.__dict__[key] = data[key]
    
    # added
    args.data_device = 'cpu'
    args.compute_cov3D_python = True
    args.convert_SHs_python = True
    # args.iterations = 10_000
    args.sh_degree = 3
    args.model_path = args.s2_model_path
    args.iterations = args.s2_iterations
    args.checkpoint_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.warm_up_iter,
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.coarse_mesh_path, args.aabb_fg_bg, 
             args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
