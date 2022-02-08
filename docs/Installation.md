## Getting started

Start by cloning the repo:

```bash
git clone git@github.com:YuliangXiu/ICON.git
cd ICON
```  

## Required packages (Ubuntu 20.04.3 LTS, CUDA=11.0)
  * Python = 3.8
  * PyTorch = 1.8.2(LTS) (official [Get Started](https://pytorch.org/get-started/locally/))
  * PyTorch3D (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))

```bash

# create conda env and install required libs
conda create --name icon python=3.8
conda activate icon
conda install -c conda-forge pyembree # used by trimesh
pip install -r requirements.txt --user
```

For data generation and training

  * freeglut (`sudo apt-get install freeglut3-dev`)
  * (optional) EGL used for headless rendering (`apt install libgl1-mesa-dri libegl1-mesa libgbm1`)

## Register the following sites
  * [SMPL Model (Male, Female)](http://smpl.is.tue.mpg.de/)
  * [SMPL Model (Neutral)](http://smplify.is.tue.mpg.de/)
  * [ICON](https://icon.is.tue.mpg.de/)
  * (optional, training used) [SMPL-X Model](http://smpl-x.is.tue.mpg.de/)
  * (optional, training used) [SMPL-(X) Kid Model](https://agora.is.tue.mpg.de/)

## Downloading required models and extra data
  ```bash
  cd ICON
  bash fetch_data.sh # requires username and password
  ```
  * Download [PyMAF](https://github.com/HongwenZhang/PyMAF#necessary-files) and [PARE(optional)](https://github.com/mkocabas/PARE#demo)
  
  ```bash
  bash fetch_hps.sh
  ```


## Tree structure of **data** folder

```
data/
├── ckpt/
│   ├── icon-filter.ckpt
│   ├── icon-nofilter.ckpt
│   ├── normal.ckpt
│   ├── pamir.ckpt
│   └── pifu.ckpt
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

