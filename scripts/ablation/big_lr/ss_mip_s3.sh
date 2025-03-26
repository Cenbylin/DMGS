#!/bin/bash
# set -e
cd "$(dirname "$0")"/../../..


python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_bicycle.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_bonsai.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_counter.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_garden.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_kitchen.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_room.json
python train_geo_stage3.py --config configs/ihpc/ablation/ss_mip_s3/mip_stump.json