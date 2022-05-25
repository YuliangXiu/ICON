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
from .head import PareHead, SMPLHead, SMPLCamHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from ..utils.train_utils import load_pretrained_model


class PARE(nn.Module):
    def __init__(
        self,
        num_joints=24,
        softmax_temp=1.0,
        num_features_smpl=64,
        backbone='resnet50',
        focal_length=5000.,
        img_res=224,
        pretrained=None,
        iterative_regression=False,
        iter_residual=False,
        num_iterations=3,
        shape_input_type='feats',  # 'feats.all_pose.shape.cam',
        # 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'
        pose_input_type='feats',
        pose_mlp_num_layers=1,
        shape_mlp_num_layers=1,
        pose_mlp_hidden_size=256,
        shape_mlp_hidden_size=256,
        use_keypoint_features_for_smpl_regression=False,
        use_heatmaps='',
        use_keypoint_attention=False,
        keypoint_attention_act='softmax',
        use_postconv_keypoint_attention=False,
        use_scale_keypoint_attention=False,
        use_final_nonlocal=None,
        use_branch_nonlocal=None,
        use_hmr_regression=False,
        use_coattention=False,
        num_coattention_iter=1,
        coattention_conv='simple',
        deconv_conv_kernel_size=4,
        use_upsampling=False,
        use_soft_attention=False,
        num_branch_iteration=0,
        branch_deeper=False,
        num_deconv_layers=3,
        num_deconv_filters=256,
        use_resnet_conv_hrnet=False,
        use_position_encodings=None,
        use_mean_camshape=False,
        use_mean_pose=False,
        init_xavier=False,
        use_cam=False,
    ):
        super(PARE, self).__init__()
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(pretrained=True,
                                           downsample=False,
                                           use_conv=(use_conv == 'conv'))
        else:
            self.backbone = eval(backbone)(pretrained=True)

        # self.backbone = eval(backbone)(pretrained=True)
        self.head = PareHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(
                backbone)['n_output_channels'],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            iterative_regression=iterative_regression,
            iter_residual=iter_residual,
            num_iterations=num_iterations,
            shape_input_type=shape_input_type,
            pose_input_type=pose_input_type,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_keypoint_features_for_smpl_regression=use_keypoint_features_for_smpl_regression,
            use_heatmaps=use_heatmaps,
            use_keypoint_attention=use_keypoint_attention,
            use_postconv_keypoint_attention=use_postconv_keypoint_attention,
            keypoint_attention_act=keypoint_attention_act,
            use_scale_keypoint_attention=use_scale_keypoint_attention,
            # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            use_branch_nonlocal=use_branch_nonlocal,
            # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            use_final_nonlocal=use_final_nonlocal,
            backbone=backbone,
            use_hmr_regression=use_hmr_regression,
            use_coattention=use_coattention,
            num_coattention_iter=num_coattention_iter,
            coattention_conv=coattention_conv,
            use_upsampling=use_upsampling,
            use_soft_attention=use_soft_attention,
            num_branch_iteration=num_branch_iteration,
            branch_deeper=branch_deeper,
            use_resnet_conv_hrnet=use_resnet_conv_hrnet,
            use_position_encodings=use_position_encodings,
            use_mean_camshape=use_mean_camshape,
            use_mean_pose=use_mean_pose,
            init_xavier=init_xavier,
        )

        self.use_cam = use_cam
        if self.use_cam:
            self.smpl = SMPLCamHead(img_res=img_res, )
        else:
            self.smpl = SMPLHead(focal_length=focal_length, img_res=img_res)

        if pretrained is not None:
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
        gt_segm=None,
    ):
        features = self.backbone(images)
        hmr_output = self.head(features, gt_segm=gt_segm)

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
                normalize_joints2d=True,
            )
            smpl_output.update(hmr_output)
        else:
            if isinstance(hmr_output['pred_pose'], list):
                # if we have multiple smpl params prediction
                # create a dictionary of lists per prediction
                smpl_output = {
                    'smpl_vertices': [],
                    'smpl_joints3d': [],
                    'smpl_joints2d': [],
                    'pred_cam_t': [],
                }
                for idx in range(len(hmr_output['pred_pose'])):
                    smpl_out = self.smpl(
                        rotmat=hmr_output['pred_pose'][idx],
                        shape=hmr_output['pred_shape'][idx],
                        cam=hmr_output['pred_cam'][idx],
                        normalize_joints2d=True,
                    )
                    for k, v in smpl_out.items():
                        smpl_output[k].append(v)
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

    # def load_backbone_pretrained(self, file):
    #     # This is usually used to load pretrained 2d keypoint detector weights
    #     logger.warning(f'Loading pretrained **backbone** weights from {file}')
    #     state_dict = torch.load(file)['model']
    #     self.backbone.load_state_dict(state_dict, strict=False)


def get_pare_model(device):
    PARE_CKPT = '/ps/scratch/ps_shared/mkocabas/pare_results/pare_pretrained_ckpt_for_smplify/epoch=14.ckpt.backup'
    PARE_CFG = '/ps/scratch/ps_shared/mkocabas/pare_results/pare_pretrained_ckpt_for_smplify/config_to_run.yaml'

    from ..core.config import get_hparams_defaults, update_hparams
    from ..utils.train_utils import load_pretrained_model
    # cfg = get_hparams_defaults()
    cfg = update_hparams(PARE_CFG)

    model_cfg = cfg
    model = PARE(
        backbone=model_cfg.PARE.BACKBONE,
        num_joints=model_cfg.PARE.NUM_JOINTS,
        softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
        num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
        focal_length=model_cfg.DATASET.FOCAL_LENGTH,
        img_res=model_cfg.DATASET.IMG_RES,
        pretrained=None,
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
    ).to(device)
    model.eval()

    logger.info(f'Loading pretrained model from {PARE_CKPT}')
    ckpt = torch.load(PARE_CKPT)['state_dict']
    load_pretrained_model(model,
                          ckpt,
                          overwrite_shape_mismatch=True,
                          remove_lightning=True)

    return model
