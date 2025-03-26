#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_chair.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_drums.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_ficus.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_hotdog.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_lego.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_materials.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_mic.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/3gs_refine/synthetic_ship.json