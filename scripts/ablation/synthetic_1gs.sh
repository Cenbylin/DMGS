#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_chair.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_drums.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_ficus.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_hotdog.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_lego.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_materials.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_mic.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine/synthetic_ship.json