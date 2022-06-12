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
import json
from yacs.config import CfgNode as CN

# Configuration variables
cfg = CN(new_allowed=True)

cfg.OUTPUT_DIR = 'results'
cfg.DEVICE = 'cuda'
cfg.DEBUG = False
cfg.LOGDIR = ''
cfg.VAL_VIS_BATCH_FREQ = 200
cfg.TRAIN_VIS_ITER_FERQ = 1000
cfg.SEED_VALUE = -1

cfg.TRAIN = CN(new_allowed=True)

cfg.LOSS = CN(new_allowed=True)
cfg.LOSS.KP_2D_W = 300.0
cfg.LOSS.KP_3D_W = 300.0
cfg.LOSS.SHAPE_W = 0.06
cfg.LOSS.POSE_W = 60.0
cfg.LOSS.VERT_W = 0.0

# Loss weights for dense correspondences
cfg.LOSS.INDEX_WEIGHTS = 2.0
# Loss weights for surface parts. (24 Parts)
cfg.LOSS.PART_WEIGHTS = 0.3
# Loss weights for UV regression.
cfg.LOSS.POINT_REGRESSION_WEIGHTS = 0.5

cfg.MODEL = CN(new_allowed=True)

cfg.MODEL.PyMAF = CN(new_allowed=True)

# switch
cfg.TRAIN.VAL_LOOP = True

cfg.TEST = CN(new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    # return cfg.clone()
    return cfg


def update_cfg(cfg_file):
    # cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    # return cfg.clone()
    return cfg


def parse_args(args):
    cfg_file = args.cfg_file
    if args.cfg_file is not None:
        cfg = update_cfg(args.cfg_file)
    else:
        cfg = get_cfg_defaults()

    # if args.misc is not None:
    #     cfg.merge_from_list(args.misc)

    return cfg


def parse_args_extend(args):
    if args.resume:
        if not os.path.exists(args.log_dir):
            raise ValueError(
                'Experiment are set to resume mode, but log directory does not exist.'
            )

        # load log's cfg
        cfg_file = os.path.join(args.log_dir, 'cfg.yaml')
        cfg = update_cfg(cfg_file)

        if args.misc is not None:
            cfg.merge_from_list(args.misc)
    else:
        parse_args(args)
