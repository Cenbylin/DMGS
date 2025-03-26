#!/bin/bash
# set -e
cd "$(dirname "$0")"/..

python -u train_geo_stage1.py -s data/nerf_synthetic/chair --model_path output/synthetic_s1/chair
python -u train_geo_stage1.py -s data/nerf_synthetic/drums --model_path output/synthetic_s1/drums
python -u train_geo_stage1.py -s data/nerf_synthetic/ficus --model_path output/synthetic_s1/ficus
python -u train_geo_stage1.py -s data/nerf_synthetic/hotdog --model_path output/synthetic_s1/hotdog
python -u train_geo_stage1.py -s data/nerf_synthetic/lego --model_path output/synthetic_s1/lego
python -u train_geo_stage1.py -s data/nerf_synthetic/materials --model_path output/synthetic_s1/materials
python -u train_geo_stage1.py -s data/nerf_synthetic/mic --model_path output/synthetic_s1/mic
python -u train_geo_stage1.py -s data/nerf_synthetic/ship --model_path output/synthetic_s1/ship

python -u train_geo_stage1.py -s data/mip-nerf360/bicycle --model_path output/mipnerf360_s1/bicycle
python -u train_geo_stage1.py -s data/mip-nerf360/bonsai --model_path output/mipnerf360_s1/bonsai
python -u train_geo_stage1.py -s data/mip-nerf360/counter --model_path output/mipnerf360_s1/counter
python -u train_geo_stage1.py -s data/mip-nerf360/garden --model_path output/mipnerf360_s1/garden
python -u train_geo_stage1.py -s data/mip-nerf360/kitchen --model_path output/mipnerf360_s1/kitchen
python -u train_geo_stage1.py -s data/mip-nerf360/room --model_path output/mipnerf360_s1/room
python -u train_geo_stage1.py -s data/mip-nerf360/stump --model_path output/mipnerf360_s1/stump