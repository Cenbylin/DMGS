#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_chair.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_drums.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_ficus.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_hotdog.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_lego.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_materials.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_mic.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale/synthetic_ship.json