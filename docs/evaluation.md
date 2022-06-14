## Evaluation

## Prerequirement

Make sure you have already generated all the required synthetic data (refer to [Dataset Instruction](dataset.md)) under `./data/thuman2_{num_views}views`, which includes the rendered RGB (`render/`), normal images(`normal_B/`, `normal_F/`, `T_normal_B/`, `T_normal_F/`), corresponding calibration matrix (`calib/`) and pre-computed visibility arrays (`vis/`).


:warning: Don't support headless mode currently, `unset PYOPENGL_PLATFORM` before training. (will fix it later...)

## Command

```bash
conda activate icon

# ICON w/ filter
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/icon-filter.yaml -test

# ICON w/o filter
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/icon-nofilter.yaml -test

# PIFu
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/pifu.yaml -test

# PaMIR
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg ./configs/train/pamir.yaml -test
```

## Intermediate Results

All the intermediate results are exported at `./results/ckpt_perfect_smpl`

## Benchmark on THuman2.0 (testset: 0500-0504, 3 views)

|Metrics|ICON w/ filter|ICON w/o filter|PIFu|PaMIR|
|---|---|---|---|---|
|Chamfer|1.068|1.218|1.711|-|
|P2S|1.068|1.210|1.159|-|
|NC|0.061|0.073|0.075|-|