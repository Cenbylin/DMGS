#!/bin/bash
# set -e
cd "$(dirname "$0")"/..

python -u train_geo_stage3.py --config configs/ihpc/mip_bicycle.json
python -u train_geo_stage3.py --config configs/ihpc/mip_bonsai.json
python -u train_geo_stage3.py --config configs/ihpc/mip_counter.json
python -u train_geo_stage3.py --config configs/ihpc/mip_garden.json
python -u train_geo_stage3.py --config configs/ihpc/mip_kitchen.json
python -u train_geo_stage3.py --config configs/ihpc/mip_room.json
python -u train_geo_stage3.py --config configs/ihpc/mip_stump.json