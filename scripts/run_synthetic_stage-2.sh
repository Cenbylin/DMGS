#!/bin/bash
# set -e
cd "$(dirname "$0")"/..

python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_chair.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_drums.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_ficus.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_hotdog.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_lego.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_materials.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_mic.json
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_ship.json