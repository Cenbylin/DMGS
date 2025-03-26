#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..


python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_bicycle.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_bonsai.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_counter.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_garden.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_kitchen.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_room.json
python -u train_geo_stage2_colmap.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_stump.json


python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_bicycle.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_bonsai.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_counter.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_garden.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_kitchen.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_room.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_learn_scale_mip/mip_stump.json