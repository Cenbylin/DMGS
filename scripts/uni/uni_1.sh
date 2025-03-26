#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_bicycle.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_bonsai.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_counter.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_garden.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_kitchen.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_room.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/uni_1/mip_stump.json