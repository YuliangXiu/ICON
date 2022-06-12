# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import time
import yaml
import shutil
import argparse
import operator
import itertools
from os.path import join
from loguru import logger
from functools import reduce
from yacs.config import CfgNode as CN
from typing import Dict, List, Union, Any
from flatten_dict import flatten, unflatten

##### CONSTANTS #####
DATASET_NPZ_PATH = 'data/dataset_extras'
DATASET_LMDB_PATH = 'data/lmdb'

PW3D_ROOT = 'data/dataset_folders/3dpw'
OH3D_ROOT = 'data/dataset_folders/3doh'

# pare
pare_data_dir = os.path.join(os.path.dirname(__file__),
                             "../../../../data/pare_data")

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(pare_data_dir,
                                           'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(pare_data_dir, 'J_regressor_h36m.npy')
SMPL_MEAN_PARAMS = os.path.join(pare_data_dir, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = os.path.join(pare_data_dir, '../smpl_related/models/smpl')
COCO_OCCLUDERS_FILE = os.path.join(
    pare_data_dir, 'occlusion_augmentation/coco_train2014_occluders.pkl')
PASCAL_OCCLUDERS_FILE = os.path.join(
    pare_data_dir, 'occlusion_augmentation/pascal_occluders.pkl')

DATASET_FOLDERS = {
    '3dpw': PW3D_ROOT,
    '3dpw-val': PW3D_ROOT,
    '3dpw-val-cam': PW3D_ROOT,
    '3dpw-test-cam': PW3D_ROOT,
    '3dpw-train-cam': PW3D_ROOT,
    '3dpw-cam': PW3D_ROOT,
    '3dpw-all': PW3D_ROOT,
    '3doh': OH3D_ROOT,
}

DATASET_FILES = [
    # Training
    {
        '3dpw-all': join(DATASET_NPZ_PATH, '3dpw_all_test_with_mmpose.npz'),
        '3doh': join(DATASET_NPZ_PATH, '3doh_test.npz'),
    },
    # Testing
    {
        '3doh': join(DATASET_NPZ_PATH, '3doh_train.npz'),
        '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
    }
]

EVAL_MESH_DATASETS = ['3dpw', '3dpw-val', '3dpw-all', '3doh']

##### CONFIGS #####
hparams = CN()

# General settings
hparams.LOG_DIR = 'logs/experiments'
hparams.METHOD = 'pare'
hparams.EXP_NAME = 'default'
hparams.RUN_TEST = False
hparams.PROJECT_NAME = 'pare'
hparams.SEED_VALUE = -1

hparams.SYSTEM = CN()
hparams.SYSTEM.GPU = ''
hparams.SYSTEM.CLUSTER_NODE = 0.0

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.LOAD_TYPE = 'Base'
hparams.DATASET.NOISE_FACTOR = 0.4
hparams.DATASET.ROT_FACTOR = 30
hparams.DATASET.SCALE_FACTOR = 0.25
hparams.DATASET.FLIP_PROB = 0.5
hparams.DATASET.CROP_PROB = 0.0
hparams.DATASET.CROP_FACTOR = 0.0
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.TRAIN_DS = 'all'
hparams.DATASET.VAL_DS = '3dpw_3doh'
hparams.DATASET.NUM_IMAGES = -1
hparams.DATASET.TRAIN_NUM_IMAGES = -1
hparams.DATASET.TEST_NUM_IMAGES = -1
hparams.DATASET.IMG_RES = 224
hparams.DATASET.USE_HEATMAPS = ''  # 'hm', 'hm_soft', 'part_segm', 'attention'
hparams.DATASET.RENDER_RES = 480
hparams.DATASET.MESH_COLOR = 'pinkish'
hparams.DATASET.FOCAL_LENGTH = 5000.
hparams.DATASET.IGNORE_3D = False
hparams.DATASET.USE_SYNTHETIC_OCCLUSION = False
hparams.DATASET.OCC_AUG_DATASET = 'pascal'
hparams.DATASET.USE_3D_CONF = False
hparams.DATASET.USE_GENDER = False
# this is a bit confusing but for the in the wild dataset ratios should be same, otherwise the code
# will be a bit verbose
hparams.DATASET.DATASETS_AND_RATIOS = 'h36m_mpii_lspet_coco_mpi-inf-3dhp_0.3_0.6_0.6_0.6_0.1'
hparams.DATASET.STAGE_DATASETS = '0+h36m_coco_0.2_0.8 2+h36m_coco_0.4_0.6'
# enable non parametric representation
hparams.DATASET.NONPARAMETRIC = False

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 0.0001  # 0.00003
hparams.OPTIMIZER.WD = 0.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 50
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.NUM_SMPLIFY_ITERS = 100  # 50
hparams.TRAINING.RUN_SMPLIFY = False
hparams.TRAINING.SMPLIFY_THRESHOLD = 100
hparams.TRAINING.DROPOUT_P = 0.2
hparams.TRAINING.TEST_BEFORE_TRAINING = False
hparams.TRAINING.SAVE_IMAGES = False
hparams.TRAINING.USE_PART_SEGM_LOSS = False
hparams.TRAINING.USE_AMP = False

# Training process hparams
hparams.TESTING = CN()
hparams.TESTING.SAVE_IMAGES = False
hparams.TESTING.SAVE_FREQ = 1
hparams.TESTING.SAVE_RESULTS = True
hparams.TESTING.SAVE_MESHES = False
hparams.TESTING.SIDEVIEW = True
hparams.TESTING.TEST_ON_TRAIN_END = True
hparams.TESTING.MULTI_SIDEVIEW = False
hparams.TESTING.USE_GT_CAM = False

# PARE method hparams
hparams.PARE = CN()
hparams.PARE.BACKBONE = 'resnet50'  # hrnet_w32-conv, hrnet_w32-interp
hparams.PARE.NUM_JOINTS = 24
hparams.PARE.SOFTMAX_TEMP = 1.
hparams.PARE.NUM_FEATURES_SMPL = 64
hparams.PARE.USE_ATTENTION = False
hparams.PARE.USE_SELF_ATTENTION = False
hparams.PARE.USE_KEYPOINT_ATTENTION = False
hparams.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION = False
hparams.PARE.USE_POSTCONV_KEYPOINT_ATTENTION = False
hparams.PARE.KEYPOINT_ATTENTION_ACT = 'softmax'
hparams.PARE.USE_SCALE_KEYPOINT_ATTENTION = False
hparams.PARE.USE_FINAL_NONLOCAL = None
hparams.PARE.USE_BRANCH_NONLOCAL = None
hparams.PARE.USE_HMR_REGRESSION = False
hparams.PARE.USE_COATTENTION = False
hparams.PARE.NUM_COATTENTION_ITER = 1
# 'double_1', 'double_3', 'single_1', 'single_3', 'simple'
hparams.PARE.COATTENTION_CONV = 'simple'
hparams.PARE.USE_UPSAMPLING = False
hparams.PARE.DECONV_CONV_KERNEL_SIZE = 4
hparams.PARE.USE_SOFT_ATTENTION = False
hparams.PARE.NUM_BRANCH_ITERATION = 0
hparams.PARE.BRANCH_DEEPER = False
hparams.PARE.NUM_DECONV_LAYERS = 3
hparams.PARE.NUM_DECONV_FILTERS = 256
hparams.PARE.USE_RESNET_CONV_HRNET = False
hparams.PARE.USE_POS_ENC = False

hparams.PARE.ITERATIVE_REGRESSION = False
hparams.PARE.ITER_RESIDUAL = False
hparams.PARE.NUM_ITERATIONS = 3
hparams.PARE.SHAPE_INPUT_TYPE = 'feats.all_pose.shape.cam'
hparams.PARE.POSE_INPUT_TYPE = 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'

hparams.PARE.POSE_MLP_NUM_LAYERS = 1
hparams.PARE.SHAPE_MLP_NUM_LAYERS = 1
hparams.PARE.POSE_MLP_HIDDEN_SIZE = 256
hparams.PARE.SHAPE_MLP_HIDDEN_SIZE = 256

hparams.PARE.SHAPE_LOSS_WEIGHT = 0
hparams.PARE.KEYPOINT_LOSS_WEIGHT = 5.
hparams.PARE.KEYPOINT_NATIVE_LOSS_WEIGHT = 5.
hparams.PARE.HEATMAPS_LOSS_WEIGHT = 5.
hparams.PARE.SMPL_PART_LOSS_WEIGHT = 1.
hparams.PARE.PART_SEGM_LOSS_WEIGHT = 1.
hparams.PARE.POSE_LOSS_WEIGHT = 1.
hparams.PARE.BETA_LOSS_WEIGHT = 0.001
hparams.PARE.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.PARE.GT_TRAIN_WEIGHT = 1.
hparams.PARE.LOSS_WEIGHT = 60.
hparams.PARE.USE_SHAPE_REG = False
hparams.PARE.USE_MEAN_CAMSHAPE = False
hparams.PARE.USE_MEAN_POSE = False
hparams.PARE.INIT_XAVIER = False


def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """
    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k, v in flattened_config_dict.items():
        if isinstance(v, list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool):
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v, list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0]
                                                  in ['True', 'False']):
                exp[param_name] = [
                    True if x == 'True' else False for x in param_value
                ]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False

        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params


def run_grid_search_experiments(
        cfg_id,
        cfg_file,
        use_cluster,
        bid,
        memory,
        script='main.py',
        cmd_opts=[],
        gpu_min_mem=10000,
        gpu_arch=('tesla', 'quadro', 'rtx'),
):
    cfg = yaml.load(open(cfg_file))
    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    logger.info(f'Grid search hparams: \n {hyperparams}')

    different_configs = [
        update_hparams_from_dict(c) for c in different_configs
    ]
    logger.info(
        f'======> Number of experiment configurations is {len(different_configs)}'
    )

    config_to_run = CN(different_configs[cfg_id])
    config_to_run.merge_from_list(cmd_opts)

    # if use_cluster:
    #     execute_task_on_cluster(
    #         script=script,
    #         exp_name=config_to_run.EXP_NAME,
    #         num_exp=len(different_configs),
    #         cfg_file=cfg_file,
    #         bid_amount=bid,
    #         num_workers=config_to_run.DATASET.NUM_WORKERS,
    #         memory=memory,
    #         exp_opts=cmd_opts,
    #         gpu_min_mem=gpu_min_mem,
    #         gpu_arch=gpu_arch,
    #     )
    #     exit()

    # ==== create logdir using hyperparam settings
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{config_to_run.EXP_NAME}'

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split('/'))
        logdir += f'_{hp.replace("/", ".").replace("_", "").lower()}-{v}'

    if script == 'occlusion_analysis.py':
        logdir = config_to_run.LOG_DIR + '/occlusion_test'
    elif script == 'eval.py':
        if config_to_run.DATASET.USE_GENDER:
            logdir = config_to_run.LOG_DIR + \
                '/evaluation_mesh_gender_' + config_to_run.DATASET.VAL_DS
        else:
            logdir = config_to_run.LOG_DIR + '/evaluation_mesh_' + config_to_run.DATASET.VAL_DS
    elif script == 'visualize_activations.py':
        logdir = config_to_run.LOG_DIR + '/vis_act'
    elif script == 'visualize_2d_heatmaps.py':
        logdir = config_to_run.LOG_DIR + '/vis_2d_heatmaps'
    elif script == 'visualize_part_segm.py':
        logdir = config_to_run.LOG_DIR + '/vis_parts'
    elif script == 'eval_cam.py':
        if config_to_run.DATASET.USE_GENDER:
            logdir = config_to_run.LOG_DIR + \
                '/evaluation_mesh_j24_gender_' + config_to_run.DATASET.VAL_DS
        else:
            logdir = config_to_run.LOG_DIR + \
                '/evaluation_mesh_j24_' + config_to_run.DATASET.VAL_DS
    else:
        logdir = os.path.join(config_to_run.LOG_DIR,
                              config_to_run.PROJECT_NAME,
                              config_to_run.EXP_NAME, logdir + '_train')

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file,
                dst=os.path.join(config_to_run.LOG_DIR, 'config.yaml'))

    config_to_run.LOG_DIR = logdir

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.LOG_DIR, 'config_to_run.yaml'))

    return config_to_run
