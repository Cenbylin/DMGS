#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..


python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_chair.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_drums.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_ficus.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_hotdog.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_lego.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_materials.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_mic.json
python -u train_geo_stage2.py --config configs/ihpc/ablation/no_learn_scale/synthetic_ship.json


python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_chair.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_drums.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_ficus.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_hotdog.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_lego.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_materials.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_mic.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale/synthetic_ship.json