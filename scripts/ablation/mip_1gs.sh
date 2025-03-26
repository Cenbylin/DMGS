#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..


python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_bicycle.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_bonsai.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_counter.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_garden.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_kitchen.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_room.json
python train_geo_stage3.py --config configs/ihpc/ablation/1gs_refine_mip/mip_stump.json