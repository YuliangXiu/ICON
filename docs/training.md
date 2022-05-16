## Training

## Prerequirement

Make sure you have already generated all the required synthetic data (refer to [Dataset Preprocess Instruction](dataset.md)) under `./data/thuman2_{num_views}views`, which includes the rendered RGB (`render/`), normal images(`normal_B/`, `normal_F/`, `T_normal_B/`, `T_normal_F/`), corresponding calibration matrix (`calib/`) and pre-computed visibility arrays (`vis/`).

:eyes: Test your dataloader with [vedo](https://vedo.embl.es/)
```bash
cd ICON/lib
python dataset/PIFudataset.py -v
```

<p align="center">
    <img src="../assets/vedo.gif" width=50%>
</p>

:warning: Don't support headless mode currently, `unset PYOPENGL_PLATFORM` before training. (will fix it later...)
## Command

```bash
conda activate icon
cd ICON/apps

# ICON w/ filter (name: icon-filter)
CUDA_VISIBLE_DEVICES=0 python train.py -cfg ../config/train/icon-filter.yaml

# ICON w/o filter (name: icon-nofilter)
CUDA_VISIBLE_DEVICES=0 python train.py -cfg ../config/train/icon-nofilter.yaml

# PIFu (name: pifu)
CUDA_VISIBLE_DEVICES=0 python train.py -cfg ../config/train/pifu.yaml
```

## Tensorboard

```bash
cd ICON/results/{name}
tensorboard --logdir .
```

## Checkpoint

All the checkpoints are saved at `./data/ckpt/{name}`