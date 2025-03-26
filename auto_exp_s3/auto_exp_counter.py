import itertools
import subprocess
import sys, os

from pathlib import Path

base_dir = str(Path(os.path.realpath(__file__)).parent.parent)

hyper_params = {
    "aabb_fg_bg": ['[[[-3, -1.0, -5], [5.0, 5, 5]], [[-12, -8, -10], [13, 13, 17]]]'],
    "res_fg": [230],
    "res_bg": [230],
    "c2f_steps": ['[2000, 3000, 4000, 5000]'],

    "sdf_lr": [0.003],
    "deform_lr": [0.02],
    "sdf_lr_bg": [0.003],
    "deform_lr_bg": [0.02],
    "final_lr_rate": [0.05],
    "texture_lr": [0.005, 0.001],
    "scale_factor_lr": [0.001],
    "lambda_sdf": ['[0.3, 0.1]'],
    "lambda_flex": ['[0.1, 0.05]']
}

# colored console output
green = lambda x: '\033[92m' + x + '\033[0m'
blue = lambda x: '\033[94m' + x + '\033[0m'

config_dir = f"{base_dir}/configs/ihpc/next_template"
with open(f"{config_dir}/mip_counter_template.json", "r") as f:
    template_s = f.read()

param_names = list(hyper_params.keys())
for id, params in enumerate(
    itertools.product(*tuple(hyper_params.values()))):
    
    out_config = template_s
    expname = f"e{id:0>3}"
    out_config = out_config.replace("$(expname)", expname)
    desc_str = expname
    for name, value in zip(param_names, params):
        out_config = out_config.replace(f"$({name})", str(value))
        desc_str += f"-{name}({str(value)})"
    
    print(green(desc_str))

    out_config_path = f"{config_dir}/_generated_{str(hash(desc_str))[-5:]}.json"
    with open(out_config_path, "w") as f:
        f.write(out_config)
    
    def execute(cmd, cwd=None):
        popen = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            print('error occur!! code:', return_code)
            # raise subprocess.CalledProcessError(return_code, cmd)

    # execute command
    for path in execute(
        ['env', 'CUDA_VISIBLE_DEVICES=0', sys.executable, '-u', 
         'train_geo_stage2_colmap.py', '--config', out_config_path], 
         cwd=base_dir):
        print(blue(path), end="")
    os.remove(out_config_path)