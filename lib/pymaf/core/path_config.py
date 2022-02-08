"""
This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/path_config.py
path configuration
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join, expanduser
import sys, os

# pymaf
pymaf_data_dir = os.path.join(os.path.dirname(__file__),
                              "../../../data/pymaf_data")

SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, '../smpl_related/models/smpl')

CUBE_PARTS_FILE = os.path.join(pymaf_data_dir, 'cube_parts.npy')
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(pymaf_data_dir,
                                           'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(pymaf_data_dir, 'J_regressor_h36m.npy')
VERTEX_TEXTURE_FILE = os.path.join(pymaf_data_dir, 'vertex_texture.npy')
SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, '../smpl_related/models/smpl')
CHECKPOINT_FILE = os.path.join(pymaf_data_dir,
                               'pretrained_model/PyMAF_model_checkpoint.pt')

# pare
pare_data_dir = os.path.join(os.path.dirname(__file__),
                             "../../../data/pare_data")
CFG = os.path.join(pare_data_dir, 'pare/checkpoints/pare_w_3dpw_config.yaml')
CKPT = os.path.join(pare_data_dir,
                    'pare/checkpoints/pare_w_3dpw_checkpoint.ckpt')
