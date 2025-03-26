#!/bin/bash
# set -e
cd "$(dirname "$0")"/..

python render_and_eval.py --config configs/ihpc/mip_bicycle.json
python render_and_eval.py --config configs/ihpc/mip_bonsai.json
python render_and_eval.py --config configs/ihpc/mip_counter.json
python render_and_eval.py --config configs/ihpc/mip_garden.json
python render_and_eval.py --config configs/ihpc/mip_kitchen.json
python render_and_eval.py --config configs/ihpc/mip_room.json
python render_and_eval.py --config configs/ihpc/mip_stump.json

# python metrics.py -m output/tmp_mipnerf360_s3/bicycle
# python metrics.py -m output/tmp_mipnerf360_s3/bonsai
# python metrics.py -m output/tmp_mipnerf360_s3/counter
# python metrics.py -m output/tmp_mipnerf360_s3/garden
# python metrics.py -m output/tmp_mipnerf360_s3/kitchen
# python metrics.py -m output/tmp_mipnerf360_s3/room
# python metrics.py -m output/tmp_mipnerf360_s3/stump