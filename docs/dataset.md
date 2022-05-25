## Environment 

  * freeglut (`sudo apt-get install freeglut3-dev`)
  * (optional) **EGL** used for headless rendering (`apt install libgl1-mesa-dri libegl1-mesa libgbm1`)

:warning: For **EGL** headless rendering (without screen, such as clusters), please `export PYOPENGL_PLATFORM=egl` before running these scripts, also set `egl=True` in `scripts/render_single.py`, otherwise, set `egl=False` and `unset PYOPENGL_PLATFORM`.

:warning: If the program runs so slowly and is stuck in `mesh.ray.intersects_any`, uninstall and reinstall `pyembree` and `trimesh`, more details in [issue #62](https://github.com/YuliangXiu/ICON/issues/62).

## THuman2.0

Please refer to [THuman2.0-Dataset](https://github.com/ytrock/THuman2.0-Dataset) to download the original scans into `data/thuman2/scans`, and its SMPL-X(male) fits into `data/thuman2/fits`. Then generate `all.txt` by `ls > ../all.txt` under `data/thuman2/scans`, which contains all the subject names (0000~0525).

:eyes: `./sample_data` contains one example of THuman2.0 and shows the data folder structure.

## Debug Mode

```bash
conda activate icon
bash scripts/render_batch.sh debug all
bash scripts/vis_batch.sh debug all
```

Then you will get the rendered samples & visibility results under `debug/`

## Generate Mode 

**1. Rendering phrase**: RGB images, normal images, calibration array. *If you need the depth maps, just set `depth=True` at `render_single.py:L56`*

```bash
conda activate icon
bash scripts/render_batch.sh gen all
```
You could check the rendering status from `log/render/thuman2-{num_views}-{size}-{part}.txt`

**2. Visibility phrase**: SMPL-X based visibility computation

```bash
bash scripts/vis_batch.sh gen all
```
You could check the visibility computing status from `log/vis/thuman2-{num_views}-{part}.txt`


:white_check_mark: NOW, you have all the synthetic dataset under `data/thuman2_{num_views}views`, which will be used for training. 

:arrow_right: NEXT, please jump to [Training Instruction](training.md) for more details.

## Examples

|<img src="../assets/rendering/080.png" width="150">|<img src="../assets/rendering/norm_F_080.png" width="150">|<img src="../assets/rendering/norm_B_080.png" width="150">|<img src="../assets/rendering/SMPL_norm_F_080.png" width="150">|<img src="../assets/rendering/SMPL_norm_B_080.png" width="150">|
|---|---|---|---|---|
|RGB Image|Normal(Front)|Normal(Back)|Normal(SMPL-X, Front)|Normal(SMPL-X, Back)|
|<img src="../assets/rendering/vis.png" width="150">|<img src="../assets/rendering/depth_F_080.png" width="150">|<img src="../assets/rendering/depth_B_080.png" width="150">|<img src="../assets/rendering/SMPL_depth_F_080.png" width="150">|<img src="../assets/rendering/SMPL_depth_B_080.png" width="150">|
|Visibility|Depth(Front)|Depth(Back)|Depth(SMPL-X, Front)|Depth(SMPL-X, Back)|

## Citation
If you use this dataset for your research, please consider citing:
```
@InProceedings{tao2021function4d,
  title={Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors},
  author={Yu, Tao and Zheng, Zerong and Guo, Kaiwen and Liu, Pengpeng and Dai, Qionghai and Liu, Yebin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR2021)},
  month={June},
  year={2021},
}
```
This `PyTorch Dataloader` benefits a lot from [MonoPortDataset](https://github.com/Project-Splinter/MonoPortDataset), so please consider citing:

```
@inproceedings{li2020monoport,
  title={Monocular Real-Time Volumetric Performance Capture},
  author={Li, Ruilong and Xiu, Yuliang and Saito, Shunsuke and Huang, Zeng and Olszewski, Kyle and Li, Hao},
  booktitle={European Conference on Computer Vision},
  pages={49--67},
  year={2020},
  organization={Springer}
}
  
@incollection{li2020monoportRTL,
  title={Volumetric human teleportation},
  author={Li, Ruilong and Olszewski, Kyle and Xiu, Yuliang and Saito, Shunsuke and Huang, Zeng and Li, Hao},
  booktitle={ACM SIGGRAPH 2020 Real-Time Live},
  pages={1--1},
  year={2020}
}
```

