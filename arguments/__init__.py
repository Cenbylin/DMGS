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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

class OptimizationParamsGeo(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.s2_iterations = 10000
        self.s3_iterations = 1000

        self.sdf_lr = 0.001
        self.deform_lr = 0.001
        self.cube_weight_lr = 0.001

        self.sdf_lr_bg = 0.0005
        self.deform_lr_bg = 0.0005
        self.cube_weight_lr_bg = 0.001
        self.scale_factor_lr = 0.0

        self.texture_lr = 0.01

        # self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        self.reg_decay_iter = -1
        self.lambda_sdf = 0.2
        self.lambda_flex = 0.25

        self.final_lr_rate = 1.0

        # marching config (temporarily place here)
        self.res_fg = 150
        self.res_bg = 200
        # (temporarily place here)
        self.gs_per_face = 6
        self.aabb_fg_bg = None
        self.warm_up_iter = -1

        self.s2_model_path = ""
        self.s3_model_path = ""
        self.c2f_steps = []
        self.c2f_rate = 2.

        # stage-3
        self.init_opacity = 0.1
        self.s3_gs_per_face = 6
        self.simplify_nface = 0
        self.vert_lr = None  # 0.0001
        self.vert_lr_final = None
        self.feature_lr = None  # 0.0025
        self.opacity_lr = None  # 0.05
        self.scaling_lr = None  # 0.005
        self.rotation_lr = None  # 0.001
        self.lambda_laplace = 0.1
        self.lambda_normal_consistency = 0.1
        self.subdivision_fg = 0
        self.subdivision_bg = 0
        self.use_frustum = True
        self.reset_opacity_steps = [1000, 2000, 3000]

        # ablation study
        self.adaptive_cov = True

        # dyn exps
        self.full_state = False
        self.s2_load_ckpt = None
        self.check_fixed_view = None

        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
