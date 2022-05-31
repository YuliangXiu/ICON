## Getting started

Start by cloning the repo:

```bash
git clone git@github.com:YuliangXiu/ICON.git
cd ICON
```  

## Environment
  * Ubuntu 20 / 18
  * **CUDA=11.0, GPU Memory > 12GB** 
  * Python = 3.8
  * PyTorch = 1.8.2 LTS (official [Get Started](https://pytorch.org/get-started/locally/))
  * PyTorch3D (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))

```bash
# install conda, skip if already have
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh

conda config --env --set always_yes true
conda update -n base -c defaults conda -y

# Note: 
# For google colab, please refer to ICON/colab.sh
# create conda env and install required libs (~20min)

cd ICON
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate icon
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

:warning: `rembg` requires the access to Google Drive, please refer to [@Yuhuoo's answer](https://github.com/YuliangXiu/ICON/issues/36#issuecomment-1141984308) if the program got stuck in `remove(buf.getvalue())`.

## Register at [ICON's website](https://icon.is.tue.mpg.de/)

![Register](../assets/register.png)
Required:
  * [SMPL](http://smpl.is.tue.mpg.de/):  SMPL Model (Male, Female)
  * [SMPLIFY](http://smplify.is.tue.mpg.de/): SMPL Model (Neutral)
  * [ICON](https://icon.is.tue.mpg.de/): pretrained models and extra data for ICON

Optional:
  * [SMPL-X](http://smpl-x.is.tue.mpg.de/): SMPL-X Model, used for training
  * [AGORA](https://agora.is.tue.mpg.de/): SMIL Kid Model, used for training
  * [PARE](https://pare.is.tue.mpg.de/): optional SMPL HPS estimator
  * [PIXIE](https://pixie.is.tue.mpg.de/): optional SMPL-X HPS estimator


:warning: Click **Register now** on all dependencies, then you can download them all with **ONE** account.

## Downloading required models and extra data
  ```bash
  cd ICON
  bash fetch_data.sh # requires username and password
  ```
  * Download [PyMAF](https://github.com/HongwenZhang/PyMAF#necessary-files), [PARE (optional, SMPL)](https://github.com/mkocabas/PARE#demo), [PIXIE (optional, SMPL-X)](https://pixie.is.tue.mpg.de/), [HybrIK (optional, SMPL)](https://github.com/Jeff-sjtu/HybrIK)
  
  ```bash
  bash fetch_hps.sh
  ```

  :eyes: If you want to support your HPS in ICON, please refer to [commit #060e265](https://github.com/YuliangXiu/ICON/commit/060e265bd253c6a34e65c9d0a5288c6d7ffaf68e) and [commit #3663704](https://github.com/YuliangXiu/ICON/commit/36637046dcbb5667cdfbee3b9c91b934d4c5dd05), then fork repo & pull request.

## Citation
:+1: Please consider citing these awesome HPS approaches

<details><summary>PyMAF, PARE, PIXIE, HybrIK, BEV</summary>

```
@inproceedings{pymaf2021,
  title={PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop},
  author={Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}

@inproceedings{Kocabas_PARE_2021,
  title = {{PARE}: Part Attention Regressor for {3D} Human Body Estimation},
  author = {Kocabas, Muhammed and Huang, Chun-Hao P. and Hilliges, Otmar and Black, Michael J.},
  booktitle = {Proc. International Conference on Computer Vision (ICCV)},
  pages = {11127--11137},
  month = oct,
  year = {2021},
  doi = {},
  month_numeric = {10}
}

@inproceedings{PIXIE:2021,
  title={Collaborative Regression of Expressive Bodies using Moderation}, 
  author={Yao Feng and Vasileios Choutas and Timo Bolkart and Dimitrios Tzionas and Michael J. Black},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}

@inproceedings{li2021hybrik,
  title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
  author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3383--3393},
  year={2021}
}

@InProceedings{BEV,
  author = {Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J},
  title = {Putting People in their Place: Monocular Regression of 3D People in Depth},
  booktitle = {CVPR},
  year = {2022}
}

@InProceedings{ROMP,
  author = {Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao},
  title = {Monocular, One-stage, Regression of Multiple 3D People},
  booktitle = {ICCV},
  year = {2021}
}

```
</details>

<br>

## Tree structure of **./data**

<details>

```
data/
├── ckpt/
│   ├── icon-filter.ckpt
│   ├── icon-nofilter.ckpt
│   ├── normal.ckpt
│   ├── pamir.ckpt
│   └── pifu.ckpt
├── hybrik_data/
│   ├── h36m_mean_beta.npy
│   ├── J_regressor_h36m.npy
│   ├── hybrik_config.yaml
│   └── pretrained_w_cam.pth
├── pare_data/
│   ├── J_regressor_{extra,h36m}.npy
│   ├── pare/
│   │   └── checkpoints/
│   │       ├── pare_checkpoint.ckpt
│   │       ├── pare_config.yaml
│   │       ├── pare_w_3dpw_checkpoint.ckpt
│   │       └── pare_w_3dpw_config.yaml
│   ├── smpl_mean_params.npz
│   └── smpl_partSegmentation_mapping.pkl
├── pixie_data/
│   ├── flame2smplx_tex_1024.npy
│   ├── MANO_SMPLX_vertex_ids.pkl
│   ├── pixie_model.tar
│   ├── SMPL-X__FLAME_vertex_ids.npy
│   ├── SMPL_X_template_FLAME_uv.obj
│   ├── smplx_extra_joints.yaml
│   ├── smplx_hand.obj
│   ├── SMPLX_NEUTRAL_2020.npz
│   ├── smplx_tex.obj
│   ├── smplx_tex.png
│   ├── SMPLX_to_J14.pkl
│   ├── uv_face_eye_mask.png
│   └── uv_face_mask.png
├── pymaf_data/
│   ├── cube_parts.npy
│   ├── gmm_08.pkl
│   ├── J_regressor_{extra,h36m}.npy
│   ├── mesh_downsampling.npz
│   ├── pretrained_model/
│   │   └── PyMAF_model_checkpoint.pt
│   ├── smpl_mean_params.npz
│   ├── UV_data/
│   │   ├── UV_Processed.mat
│   │   └── UV_symmetry_transforms.mat
│   └── vertex_texture.npy
├── smpl_related/
│   ├── models/
│   │   ├── smpl/
│   │   │   ├── SMPL_{FEMALE,MALE,NEUTRAL}.pkl
│   │   │   ├── smpl_kid_template.npy
│   │   └── smplx/
│   │       ├── SMPLX_{FEMALE,MALE,NEUTRAL}.npz
│   │       ├── SMPLX_{FEMALE,MALE,NEUTRAL}.pkl
│   │       ├── smplx_kid_template.npy
│   │       └── version.txt
│   └── smpl_data/
│       ├── smpl_verts.npy
│       ├── smplx_cmap.npy
│       ├── smplx_faces.npy
│       └── smplx_verts.npy
└── tedra_data/
    ├── faces.txt
    ├── tetrahedrons.txt
    ├── tetgen_{male,female,neutral}_{adult,kid}_structure.npy
    ├── tetgen_{male,female,neutral}_{adult,kid}_vertices.npy
    ├── tetra_{male,female,neutral}_{adult,kid}_smpl.npz
    ├── tetrahedrons_{male,female,neutral}_{adult,kid}.txt
    └── vertices.txt
```
</details>
