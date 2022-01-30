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

from .keypoints import JointsMSELoss
from .segmentation import CrossEntropy
from ..utils.geometry import batch_rodrigues, rotmat_to_rot6d


class PARELoss(nn.Module):
    def __init__(
        self,
        shape_loss_weight=0,
        keypoint_loss_weight=5.,
        keypoint_native_loss_weight=5.,
        heatmaps_loss_weight=1.,
        part_loss_weight=1.,
        smpl_part_loss_weight=1.,
        pose_loss_weight=1.,
        beta_loss_weight=0.001,
        openpose_train_weight=0.,
        gt_train_weight=1.,
        loss_weight=60.,
        use_heatmaps_loss=False,
        use_shape_regularization=False,
    ):
        super(PARELoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_heatmaps = JointsMSELoss(
            use_target_weight=True)  # nn.MSELoss(reduction='none')
        self.criterion_segm_mask = CrossEntropy()
        self.criterion_regr = nn.MSELoss(reduction='none')
        self.criterion_part = nn.MSELoss()

        self.use_heatmaps_loss = use_heatmaps_loss
        self.use_shape_regularization = use_shape_regularization

        # self.mean_betas = torch.ones(1, 10) * 0.5
        self.mean_betas = torch.tensor(
            [0.5, 0.5, -0.5, 0.5, -0.35, 0, 0, 0.4, 0, 0.25])

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.keypoint_native_loss_weight = keypoint_native_loss_weight
        self.heatmaps_loss_weight = heatmaps_loss_weight
        self.part_loss_weight = part_loss_weight
        self.smpl_part_loss_weight = smpl_part_loss_weight

    def set_part_loss_weight(self, value):
        self.part_loss_weight = value

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_pose_conf = gt['pose_conf']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            gt_pose_conf,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        loss_heatmaps = None
        loss_keypoints_native = None
        loss_segm_mask = None
        if self.use_heatmaps_loss == 'hm':
            pred_heatmaps_2d = pred['pred_heatmaps_2d']
            gt_heatmaps_2d = gt['heatmaps_2d']
            gt_joint_vis = gt['joint_vis']

            loss_heatmaps = heatmap_2d_loss(
                pred_heatmaps_2d=pred_heatmaps_2d,
                gt_heatmaps_2d=gt_heatmaps_2d,
                joint_vis=gt_joint_vis,
                criterion=self.criterion_heatmaps,
            )
        elif self.use_heatmaps_loss == 'hm_soft':
            pred_heatmaps_2d = pred['pred_heatmaps_2d']
            pred_keypoints_2d = pred['pred_kp2d']
            gt_heatmaps_2d = gt['heatmaps_2d']
            gt_joint_vis = gt['joint_vis']

            loss_heatmaps = heatmap_2d_loss(
                pred_heatmaps_2d=pred_heatmaps_2d,
                gt_heatmaps_2d=gt_heatmaps_2d,
                joint_vis=gt_joint_vis,
                criterion=self.criterion_heatmaps,
            )
            loss_keypoints_native = keypoint_loss(
                pred_keypoints_2d,
                gt['smpl_keypoints'],
                criterion=self.criterion_keypoints)
            # clip native kp loss to avoid explosion
            loss_keypoints_native = torch.where(
                loss_keypoints_native < 10., loss_keypoints_native,
                torch.zeros_like(loss_keypoints_native))
        elif self.use_heatmaps_loss == 'part_segm':
            pred_segm_mask = pred['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt['gt_segm_mask'][has_smpl == 1]
            loss_segm_mask = self.criterion_segm_mask(score=pred_segm_mask,
                                                      target=gt_segm_mask)
        elif self.use_heatmaps_loss == 'part_segm_pool':
            pred_segm_mask = pred['pred_segm_mask'][has_smpl == 1]
            gt_segm_mask = gt['gt_segm_mask'][has_smpl == 1]
            loss_segm_mask = self.criterion_segm_mask(score=pred_segm_mask,
                                                      target=gt_segm_mask)
        elif self.use_heatmaps_loss == 'attention':
            pass
        else:
            pred_keypoints_2d = pred['pred_kp2d']
            loss_keypoints_native = keypoint_loss(
                pred_keypoints_2d,
                gt['smpl_keypoints'],
                criterion=self.criterion_keypoints)

            # clip native kp loss to avoid explosion
            loss_keypoints_native = torch.where(
                loss_keypoints_native < 10., loss_keypoints_native,
                torch.zeros_like(loss_keypoints_native))

            # loss_keypoints_native = torch.clamp(loss_keypoints_native, max=10.)

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight

        if self.use_heatmaps_loss == 'part_segm_pool':
            loss_cam = 0
        else:
            loss_cam = ((torch.exp(-pred_cam[:, 0] * 10))**2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        if loss_keypoints_native is not None:
            loss_keypoints_native *= self.keypoint_native_loss_weight
            loss_dict['loss/loss_keypoints_native'] = loss_keypoints_native

        if loss_heatmaps is not None:
            loss_heatmaps *= self.heatmaps_loss_weight
            loss_dict['loss/loss_heatmaps'] = loss_heatmaps

        if loss_segm_mask is not None:
            loss_segm_mask *= self.part_loss_weight
            loss_dict['loss/loss_segm_mask'] = loss_segm_mask

        if 'pred_segm_rgb' in pred.keys():
            loss_part_segm = self.criterion_part(pred['pred_segm_rgb'],
                                                 gt['gt_segm_rgb'])
            loss_part_segm *= self.smpl_part_loss_weight
            loss_dict['loss/loss_part_segm'] = loss_part_segm

        if self.use_shape_regularization:
            batch_size = pred_betas.shape[0]
            if self.mean_betas.device != pred_betas.device:
                self.mean_betas = self.mean_betas.to(pred_betas.device).type(
                    pred_betas.dtype)
            loss_beta_reg = (
                pred_betas -
                self.mean_betas.repeat(batch_size, 1)).pow(2).mean()
            loss_dict['loss/loss_beta_req'] = loss_beta_reg

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        for k, v in loss_dict.items():
            if 'loss_cam' in k:
                continue
            if torch.any(torch.isnan(v)):
                logger.debug(f'{k} is Nan!')
            if torch.any(torch.isinf(v)):
                logger.debug(f'{k} is Inf!')

        return loss, loss_dict


class HMRLoss(nn.Module):
    def __init__(
        self,
        shape_loss_weight=0,
        keypoint_loss_weight=5.,
        pose_loss_weight=1.,
        smpl_part_loss_weight=1.,
        beta_loss_weight=0.001,
        openpose_train_weight=0.,
        gt_train_weight=1.,
        loss_weight=60.,
        estimate_var=False,
        uncertainty_loss='MultivariateGaussianNegativeLogLikelihood',
    ):
        super(HMRLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.estimate_var = estimate_var

        if self.estimate_var:
            self.criterion_regr = eval(uncertainty_loss)()  # AleatoricLoss
        else:
            self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.smpl_part_loss_weight = smpl_part_loss_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape_var'] if self.estimate_var else pred[
            'pred_shape']
        pred_rotmat = pred['pred_pose_var'] if self.estimate_var else pred[
            'pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_pose_conf = gt['pose_conf']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        smpl_loss_f = smpl_losses_uncertainty if self.estimate_var else smpl_losses

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_loss_f(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            gt_pose_conf,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10))**2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        if 'pred_segm_rgb' in pred.keys():
            loss_part_segm = self.criterion_part(pred['pred_segm_rgb'],
                                                 gt['gt_segm_rgb'])
            loss_part_segm *= self.smpl_part_loss_weight
            loss_dict['loss/loss_part_segm'] = loss_part_segm

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict


def projected_keypoint_loss(
    pred_keypoints_2d,
    gt_keypoints_2d,
    openpose_weight,
    gt_weight,
    criterion,
    reduce='mean',
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    if reduce == 'mean':
        loss = (
            conf *
            criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    elif reduce == 'none':
        loss = (conf *
                criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1]))
    else:
        raise ValueError(f'{reduce} value is not defined!')
    return loss


def keypoint_loss(
    pred_keypoints_2d,
    gt_keypoints_2d,
    criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss


def heatmap_2d_loss(
    pred_heatmaps_2d,
    gt_heatmaps_2d,
    joint_vis,
    criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_heatmaps_2d, gt_heatmaps_2d, joint_vis)
    return loss


def keypoint_3d_loss(
    pred_keypoints_3d,
    gt_keypoints_3d,
    has_pose_3d,
    criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2, :] +
                       pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def np_keypoint_3d_loss(
    pred_joints,
    gt_joints,
    has_pose_3d,
    criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    # pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()

    # gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    # conf = conf[has_pose_3d == 1]
    # pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

    # gt_pelvis = (gt_joints[:, 2, :] + gt_joints[:, 3, :]) / 2
    # gt_keypoints_3d = gt_joints - gt_pelvis[:, None, :]
    # pred_pelvis = (pred_joints[:, 2, :] + pred_joints[:, 3, :]) / 2
    # pred_keypoints_3d = pred_joints - pred_pelvis[:, None, :]
    return criterion(pred_joints, gt_joints).mean()


def shape_loss(
    pred_vertices,
    gt_vertices,
    has_smpl,
    criterion,
):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses_uncertainty(
    pred_rot6d,
    pred_betas,
    gt_pose,
    gt_betas,
    has_smpl,
    criterion,
):
    pred_rot6d_valid = pred_rot6d[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,
                                                   3)).view(-1, 24, 3,
                                                            3)[has_smpl == 1]
    gt_rot6d_valid = rotmat_to_rot6d(gt_rotmat_valid)
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rot6d_valid) > 0:
        loss_regr_pose = criterion(pred_rot6d_valid, gt_rot6d_valid)
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rot6d.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rot6d.device)
    return loss_regr_pose, loss_regr_betas


def smpl_losses(
    pred_rotmat,
    pred_betas,
    gt_pose,
    gt_betas,
    has_smpl,
    pose_conf,
    criterion,
):
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,
                                                   3)).view(-1, 24, 3,
                                                            3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    pose_conf = pose_conf[has_smpl == 1].unsqueeze(-1).unsqueeze(-1)
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (
            pose_conf * criterion(pred_rotmat_valid, gt_rotmat_valid)).mean()
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid).mean()
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas
