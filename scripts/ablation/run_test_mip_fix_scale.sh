#!/bin/bash
# set -e
cd "$(dirname "$0")"/../..

python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_bicycle.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_bonsai.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_counter.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_garden.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_kitchen.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_room.json
python render_and_eval.py --skip_train --config configs/ihpc/ablation/no_learn_scale_mip/mip_stump.json