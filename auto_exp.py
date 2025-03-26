import itertools
import subprocess
import sys, os

base_dir = os.path.dirname(os.path.realpath(__file__))

hyper_params = {
    'warm_up_iter': [500, 1000, 2000],
    # learning rate
    'sdf_lr': [0.005],
    'deform_lr': [0.02],
    'sdf_lr_bg': [0.005],
    'deform_lr_bg': [0.02],
    'texture_lr': [0.005],
    # model
    'lambda_sdf': [0.1, 0.05],
    'lambda_flex': [0.1, 0.05],
}

# colored console output
green = lambda x: '\033[92m' + x + '\033[0m'
blue = lambda x: '\033[94m' + x + '\033[0m'

config_dir = f"{base_dir}/configs/ihpc"
with open(f"{config_dir}/mip_kitchen_template.json", "r") as f:
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