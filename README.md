### Direct Learning of Mesh and Appearance via 3D Gaussian
Splatting
- [ ] Clean codes thoroughly.
- [x] Codes realeased.


### Installation
Requires Python 3.7+, Cuda 11.3+ and PyTorch 1.10+

In addition to the dependencies in original 3DGS, you are required to install the following libraries:
```shell
pip install trimesh open3d
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Data
We assume that the `NeRF-Synthetic` and `Mip-NeRF360` datasets are located in the following directories.
```shell
ln -s /data/datasets/nerf/nerf_synthetic ./data/nerf_synthetic
ln -s /data/datasets/nerf/mip-nerf360 ./data/mip-nerf360
```

### Quick Start
P.S. We will combine these scripts in the future to support one-click start.
```shell
# for synthetic dataset
python -u train_geo_stage1.py -s data/nerf_synthetic/chair --model_path output/synthetic_s1/chair
# for mip-nerf360(colmap) dataset
python -u train_geo_stage1.py -s data/mip-nerf360/bicycle --model_path output/mipnerf360_s1/bicycle
```
Follow `train_geo_stage1_post.ipynb` to initialize the next-stage training. And then
```shell
# for synthetic dataset
python -u train_geo_stage2.py --config configs/ihpc/synthetic/synthetic_chair.json
# for mip-nerf360(colmap) dataset
python -u train_geo_stage2_colmap.py --config configs/ihpc/mip_bicycle.json
```

Below is the (optional) refinement stage.
```shell
# for synthetic dataset
python -u train_geo_stage3.py --config configs/ihpc/synthetic/synthetic_chair.json
# for mip-nerf360(colmap) dataset
python -u train_geo_stage3.py --config configs/ihpc/mip_bicycle.json
```