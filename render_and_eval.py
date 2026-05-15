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
import json
from scene import Scene
import sys
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_gsinfo
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.gs_geo_fine_unified import GaussianGeoModel
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from omegaconf import OmegaConf

@torch.no_grad()
def render_set(model_path, name, iteration, views, cam_info, gs_info, background, do_metric=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # render_white_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_w")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(render_white_path, exist_ok=True)
    ssims = []
    psnrs = []
    lpipss = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        rendering = render_gsinfo(view, gs_info, args, background, image_only=True)["render"]
        gt = view.original_image[0:3, :, :].cuda()
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # if cam_info[idx].image_alpha is not None:
            # image_alpha = torch.from_numpy(cam_info[idx].image_alpha).to('cuda').view_as(rendering[:1, ...])
            # rendering_white = rendering * image_alpha + (1-image_alpha)
            # torchvision.utils.save_image(rendering_white, os.path.join(render_white_path, '{0:05d}'.format(idx) + "_w.png"))
        
        if do_metric:
            ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))
            psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
            lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

            per_view_dict.update({
                idx: {
                    'psnr': psnrs[-1].item(),
                    'ssim': ssims[-1].item(),
                    'lpips': lpipss[-1].item(),
                }
            })

    if do_metric:
        full_dict = {
            'psnr': torch.tensor(psnrs).mean().item(),
            'ssim': torch.tensor(ssims).mean().item(),
            'lpips': torch.tensor(lpipss).mean().item(),
        }
        with open(model_path + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(model_path + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


def render_sets(args, skip_train : bool, skip_test : bool, s3_chkpnt_path):
    with torch.no_grad():
        gaussians = GaussianGeoModel(args.sh_degree)
        scene = Scene(args, gaussians, shuffle=False)

        (model_params, _) = torch.load(s3_chkpnt_path)
        gs_info = gaussians.load_for_eval(model_params)

        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(args.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene.scene_info.train_cameras, gs_info, background)

        if not skip_test:
             render_set(args.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene.scene_info.test_cameras, gs_info, background, 
                        do_metric=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)

    args, _ = parser.parse_known_args(sys.argv[1:])
    vanilla_gs_config = OmegaConf.create(vars(args))
    config = OmegaConf.load(args.config)
    if 'base_config' in config:
        base_path = os.path.join(os.path.dirname(args.config), config['base_config'])
        if os.path.exists(base_path):
            config = OmegaConf.merge(OmegaConf.load(base_path), config)
    cli = OmegaConf.from_cli()
    args = OmegaConf.merge(vanilla_gs_config, config, cli)

    args.compute_cov3D_python = False
    args.convert_SHs_python = False
    args.sh_degree = 3
    args.model_path = args.s3_model_path
    args.iterations = args.s3_iterations
    s3_chkpnt_path = f"{args.s3_model_path}/chkpnt{args.s3_iterations}.pth"

    args.images = "images"  # follow raw 3DGS, `images` for evaluation, even using images2/4 for training.

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    render_sets(args, args.skip_train, args.skip_test, s3_chkpnt_path)