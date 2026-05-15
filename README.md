# Direct Learning of Mesh and Appearance via 3D Gaussian Splatting

Paper: https://arxiv.org/abs/2405.06945

This repository contains the training code for learning mesh geometry and appearance from
3D Gaussian Splatting, plus the newer shape-adaptation stage.

- [ ] Clean codes thoroughly.
- [x] Codes realeased.

## Installation

The code was developed with Python 3.8, CUDA 11.3, and PyTorch 1.12.1. Start from the
provided environment and install the extra geometry/rendering dependencies:

```shell
conda env create -f environment.yml
conda activate gaussian_splatting

pip install trimesh open3d point-cloud-utils omegaconf kornia
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

As in 3DGS, the CUDA extensions used by `gaussian_renderer` must also be installed
(`diff-gaussian-rasterization`, `simple-knn`, and the imported surfel rasterizer module
`diff_surfel_rasterization`).

## Data

The loaders support Blender/NeRF-Synthetic scenes with `transforms_train.json` and COLMAP
scenes with `sparse/0`. The default configs expect:

```shell
ln -s /path/to/nerf_synthetic ./data/nerf_synthetic
ln -s /path/to/mip-nerf360 ./data/mip-nerf360
ln -s /path/to/dmgs_adapt ./data/dmgs_adapt
```

For adaptation, `data/dmgs_adapt/${shape_name}` is the source shape and
`data/dmgs_adapt/${adapt_shape_name}` is the target shape.

## Training

Current runs are config-driven. Edit or copy a YAML file under `configs/synthetic`,
`configs/mip`, or `configs/adapt`; important fields are `shape_name`, `images`,
`aabb_fg`, `res_fg`, and, for adaptation, `adapt_shape_name`.

Run all normal stages for one scene:

```shell
python train_unified.py --config configs/synthetic/synthetic_lego.yaml --stages 1 2 3
python train_unified.py --config configs/mip/mip_bicycle.yaml --stages 1 2 3
```

Stage meanings:

- `1`: train coarse 3DGS and build `coarse_mesh.obj`.
- `2`: learn mesh/SDF and appearance from the coarse output.
- `2a`: adapt a trained source model to a target scene.
- `3`: refine appearance on the learned mesh.

You can run a subset if previous outputs already exist:

```shell
python train_unified.py --config configs/mip/mip_bicycle.yaml --stages 2
python train_unified.py --config configs/mip/mip_bicycle.yaml --stages 3
```

The older per-stage entry points still exist and are what the batch scripts wrap:

```shell
python train_geo_stage2_unified.py --config configs/synthetic/synthetic_lego.yaml
python train_geo_stage3_unified.py --config configs/mip/mip_bicycle.yaml
```

Batch examples:

```shell
bash run_all_s2.sh
bash run_all_s3.sh
```

These scripts assume that the previous stage outputs already exist at the paths defined
in the YAML configs. For stage 1, prefer `train_unified.py --stages 1`, because it also
creates the coarse mesh needed by stage 2.

## Adaptation

First train the source shape through stage 2, then run stage `2a`:

```shell
python train_unified.py --config configs/adapt/adapt_lego.yaml --stages 1 2
python train_unified.py --config configs/adapt/adapt_lego.yaml --stages 2a
```

`2a` loads `output/adapt_s2/${shape_name}/chkpnt${s2_iterations}.pth` and
`chkpnt${s2_iterations}_sdf.pth`, then writes to
`output/adapt_s2a/${shape_name}_to_${adapt_shape_name}`.

For the prepared BOP adaptation experiments:

```shell
bash run_adapt.sh
```

## Rendering and Evaluation

After stage 3, render and compute metrics with:

```shell
python render_and_eval.py --config configs/synthetic/synthetic_lego.yaml
python render_and_eval.py --config configs/mip/mip_bicycle.yaml
```

For all listed synthetic and Mip-NeRF360 configs:

```shell
bash run_all_eval.sh
```

Outputs are written under the stage paths defined by the YAML files, for example
`output/synthetic_s1`, `output/synthetic_s2`, `output/synthetic_s3`, and their
Mip-NeRF360/adaptation counterparts.
