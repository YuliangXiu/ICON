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
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import HMRHead, SMPLHead, SMPLCamHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from ..utils.train_utils import load_pretrained_model


class HMR(nn.Module):
    def __init__(
        self,
        backbone='resnet50',
        focal_length=5000.,
        img_res=224,
        pretrained=None,
        use_cam=False,
        p=0.0,
        estimate_var=False,
        use_separate_var_branch=False,
        uncertainty_activation='',
        use_cam_feats=False,
    ):
        super(HMR, self).__init__()

        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(pretrained=True,
                                           downsample=True,
                                           use_conv=(use_conv == 'conv'))
        else:
            self.backbone = eval(backbone)(pretrained=True)

        self.use_cam_feats = use_cam_feats

        self.head = HMRHead(
            num_input_features=get_backbone_info(backbone)
            ['n_output_channels'],
            estimate_var=estimate_var,
            use_separate_var_branch=use_separate_var_branch,
            uncertainty_activation=uncertainty_activation,
            backbone=backbone,
            use_cam_feats=use_cam_feats,
        )

        self.use_cam = use_cam

        if self.use_cam:
            self.smpl = SMPLCamHead(img_res=img_res)
        else:
            self.smpl = SMPLHead(focal_length=focal_length, img_res=img_res)

        if pretrained is not None:
            if pretrained == 'data/model_checkpoint.pt':
                self.load_pretrained_spin(pretrained)
            else:
                self.load_pretrained(pretrained)

    def forward(
        self,
        images,
        cam_rotmat=None,
        cam_intrinsics=None,
        bbox_scale=None,
        bbox_center=None,
        img_w=None,
        img_h=None,
    ):
        features = self.backbone(images)

        if self.use_cam_feats:
            cam_vfov = 2 * torch.atan(img_h / (2 * cam_intrinsics[:, 0, 0]))
            hmr_output = self.head(features,
                                   cam_rotmat=cam_rotmat,
                                   cam_vfov=cam_vfov)
        else:
            hmr_output = self.head(features)

        if self.use_cam:
            smpl_output = self.smpl(
                rotmat=hmr_output['pred_pose'],
                shape=hmr_output['pred_shape'],
                cam=hmr_output['pred_cam'],
                cam_rotmat=cam_rotmat,
                cam_intrinsics=cam_intrinsics,
                bbox_scale=bbox_scale,
                bbox_center=bbox_center,
                img_w=img_w,
                img_h=img_h,
                normalize_joints2d=False,
            )
            smpl_output.update(hmr_output)
        else:
            smpl_output = self.smpl(
                rotmat=hmr_output['pred_pose'],
                shape=hmr_output['pred_shape'],
                cam=hmr_output['pred_cam'],
                normalize_joints2d=True,
            )
            smpl_output.update(hmr_output)
        return smpl_output

    def load_pretrained(self, file):
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)
        self.backbone.load_state_dict(state_dict, strict=False)
        load_pretrained_model(self.head,
                              state_dict=state_dict,
                              strict=False,
                              overwrite_shape_mismatch=True)

    def load_pretrained_spin(self, file):
        # file = '/ps/scratch/mkocabas/developments/SPIN/logs/h36m_training/checkpoints/2020_06_28-11_14_46.pt'
        # file = 'data/model_checkpoint.pt'
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['model']
        self.backbone.load_state_dict(state_dict, strict=False)
        self.head.load_state_dict(state_dict, strict=False)
