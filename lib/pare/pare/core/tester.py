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


import torch
from loguru import logger

from ..models import PARE
from .config import update_hparams
from ..utils.train_utils import load_pretrained_model

MIN_NUM_FRAMES = 0


class PARETester:
    def __init__(self, cfg, ckpt):
        self.model_cfg = update_hparams(cfg)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
        self._load_pretrained_model(ckpt)
        self.model.eval()

    def _build_model(self):
        # ========= Define PARE model ========= #
        model_cfg = self.model_cfg

        if model_cfg.METHOD == 'pare':
            model = PARE(
                backbone=model_cfg.PARE.BACKBONE,
                num_joints=model_cfg.PARE.NUM_JOINTS,
                softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
                num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
                focal_length=model_cfg.DATASET.FOCAL_LENGTH,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=model_cfg.TRAINING.PRETRAINED,
                iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
                num_iterations=model_cfg.PARE.NUM_ITERATIONS,
                iter_residual=model_cfg.PARE.ITER_RESIDUAL,
                shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
                pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
                pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
                shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
                pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
                shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
                use_keypoint_features_for_smpl_regression=model_cfg.PARE.
                USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
                use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
                use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
                use_postconv_keypoint_attention=model_cfg.PARE.
                USE_POSTCONV_KEYPOINT_ATTENTION,
                use_scale_keypoint_attention=model_cfg.PARE.
                USE_SCALE_KEYPOINT_ATTENTION,
                keypoint_attention_act=model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
                use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
                use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
                use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
                use_coattention=model_cfg.PARE.USE_COATTENTION,
                num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
                coattention_conv=model_cfg.PARE.COATTENTION_CONV,
                use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
                deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
                use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
                num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
                branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
                num_deconv_layers=model_cfg.PARE.NUM_DECONV_LAYERS,
                num_deconv_filters=model_cfg.PARE.NUM_DECONV_FILTERS,
                use_resnet_conv_hrnet=model_cfg.PARE.USE_RESNET_CONV_HRNET,
                use_position_encodings=model_cfg.PARE.USE_POS_ENC,
                use_mean_camshape=model_cfg.PARE.USE_MEAN_CAMSHAPE,
                use_mean_pose=model_cfg.PARE.USE_MEAN_POSE,
                init_xavier=model_cfg.PARE.INIT_XAVIER,
            ).to(self.device)
        else:
            logger.error(f'{model_cfg.METHOD} is undefined!')
            exit()

        return model

    def _load_pretrained_model(self, ckpt_path):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {ckpt_path}')
        ckpt = torch.load(ckpt_path)['state_dict']
        load_pretrained_model(self.model,
                              ckpt,
                              overwrite_shape_mismatch=True,
                              remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{ckpt_path}\"')
