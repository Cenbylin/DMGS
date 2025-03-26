#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_bicycle.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_bonsai.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_counter.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_garden.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_kitchen.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_room.json
python -u train_geo_stage3.py --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_stump.json

python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_bicycle.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_bonsai.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_counter.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_garden.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_kitchen.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_room.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_adaptive_cov_mip_3gs/mip_stump.json