## Environment 

  * freeglut (`sudo apt-get install freeglut3-dev`)
  * (optional) EGL used for headless rendering (`apt install libgl1-mesa-dri libegl1-mesa libgbm1`)

## THuman2.0

Please refer to [THuman2.0-Dataset](https://github.com/ytrock/THuman2.0-Dataset) to download the original scans into `data/thuman2/scans`, and its SMPL-X fits into `data/thuman2/fits`. Then generate `all.txt` by `ls > ../all.txt` under `data/thuman2/scans`, which contains all the subject names (0000~0525).

## Debug Mode

:warning: For headless rendering (without window, such as clusters), please `export PYOPENGL_PLATFORM=egl` before running these scripts, also change `egl=True` in `scripts/render_single.py`.

```bash
conda activate icon

cd ICON/scripts
bash render_batch.sh debug all
```

Then you will get the rendered samples under `debug/`

## Generate Mode 


```bash
conda activate icon
cd ICON/scripts
bash render_batch.sh gen all
```

Then you will get the whole generated dataset under `data/thuman2_{num_views}views`

## Examples

|<img src="assets/../../assets/rendering/080.png" width="150">|<img src="assets/../../assets/rendering/norm_F_080.png" width="150">|<img src="assets/../../assets/rendering/norm_B_080.png" width="150">|<img src="assets/../../assets/rendering/SMPL_norm_F_080.png" width="150">|<img src="assets/../../assets/rendering/SMPL_norm_B_080.png" width="150">|
|---|---|---|---|---|
|Image|Normal(Front)|Normal(Back)|Normal(SMPL, Front)|Normal(SMPL, Back)|

