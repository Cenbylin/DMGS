#!/bin/bash
# set -e
cd "$(dirname "$0")"/..

python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_chair.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_drums.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_ficus.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_hotdog.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_lego.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_materials.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_mic.json
python render_and_eval.py --skip_train --config configs/ihpc/synthetic/synthetic_ship.json