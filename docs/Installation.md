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
pip install -r requirements.txt

# install rembg
GPU=1 pip install rembg --upgrade

# install neural_renderer
# cuda >= 11.0
git clone https://github.com/adambielski/neural_renderer.git
cd neural_renderer && python setup.py install
# cuda < 11.0
pip install neural_renderer_pytorch

# install smplx (modified by @Yuliang Xiu)
git clone git@github.com:YuliangXiu/smplx.git
cd smplx && python setup.py install

# install bvh-distance-queries
cd lib/bvh-distance-queries
git clone https://github.com/NVIDIA/cuda-samples.git
export CUDA_SAMPLES_INC=${directory of cuda-samples/Common}
python setup.py install
```

For data generation and training

  * trimesh with pyembree (`conda install -c conda-forge pyembree`)
  * freeglut (`sudo apt-get install freeglut3-dev`)
  * (optional) EGL used for headless rendering (`apt install libgl1-mesa-dri libegl1-mesa libgbm1`)

## Pre-trained models and extra data
  * Download [SMPL Model (Male, Female)](http://smpl.is.tue.mpg.de/), Login or Register 
    * Choose `SMPL (10 shape PCs)`
    * Put SMPL models (*.pkl) under `./data/smpl_related/models/smpl`
    * Rename them as `SMPL_{FEMALE,MALE}.pkl`
  * Download [SMPL Model (Neutral)](http://smplify.is.tue.mpg.de/), Login or Register 
    * Choose `SMPLIFY_CODE_V2.ZIP`
    * Put SMPL models (*.pkl) under `./data/smpl_related/models/smpl`
    * Rename it as `SMPL_NEUTRAL.pkl`
  * Download [ICON](https://icon.is.tue.mpg.de/), Login or Register 
  ```bash
  cd icon
  bash fetch_data.sh # requires username and password
  cd data && unzip icon_data.zip
  mv smpl_data smpl_related/
  ```
  * Download [PyMAF](https://github.com/HongwenZhang/PyMAF#necessary-files) and [PARE](https://github.com/mkocabas/PARE#demo)
  
```bash
bash fetch_hps.sh
```
  * (optional) Download [SMPL-X Model](http://smpl-x.is.tue.mpg.de/), Login or Register 
    * Choose `SMPL-X v1.1 (830MB)`
    * Put SMPL-X models (*.pkl, *.npz) under `./data/smpl_related/models/smplx`  
  * (optional, used for training) Download [SMPL-(X) Kid Model](https://agora.is.tue.mpg.de/), Login or Register 
    * Put `SMIL (SMPL format)` to `./data/smpl_related/models/smpl`
    * Put `SMIL (SMPL-X format)` to `./data/smpl_related/models/smplx`


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

