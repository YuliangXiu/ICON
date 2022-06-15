## Evaluation

## Prerequirement

Make sure you have already generated all the required synthetic data (refer to [Dataset Instruction](dataset.md)) under `./data/thuman2_{num_views}views`, which includes the rendered RGB (`render/`), normal images(`normal_B/`, `normal_F/`, `T_normal_B/`, `T_normal_F/`), corresponding calibration matrix (`calib/`) and pre-computed visibility arrays (`vis/`).

:warning: Don't support headless mode currently, `unset PYOPENGL_PLATFORM` before training.

## Command

```bash
conda activate icon

# ICON w/ filter
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/icon-filter.yaml -test

# ICON w/o filter
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/icon-nofilter.yaml -test

# PIFu* (*: re-implementation)
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/pifu.yaml -test

# PaMIR* (*: re-implementation)
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/pamir.yaml -test
```

The qualitative results are located at `./results/`