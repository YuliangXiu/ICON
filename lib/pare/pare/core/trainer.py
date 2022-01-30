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
import cv2
import json
import torch
import joblib
import numpy as np
from loguru import logger
# import neural_renderer as nr
import pytorch_lightning as pl
from smplx import SMPL as SMPL_native
from torch.utils.data import DataLoader

from . import config
from . import constants
from ..models import SMPL
from ..utils.renderer import Renderer
from ..dataset import EFTDataset, BaseDataset
from ..utils.vis_utils import color_vertices_batch
from ..utils.train_utils import set_seed
from ..utils.eval_utils import reconstruction_error, compute_error_verts
from ..utils.geometry import estimate_translation, \
    perspective_projection, convert_weak_perspective_to_perspective
from ..utils.image_utils import denormalize_images, generate_heatmaps_2d_batch, generate_part_labels, \
    get_body_part_texture, get_default_camera


class PARETrainer(pl.LightningModule):
    def __init__(self, hparams):
        super(PARETrainer, self).__init__()

        self.hparams.update(hparams)

        # create networks
        if self.hparams.METHOD == 'pare':
            from ..models import PARE
            from ..losses import PARELoss
            self.model = PARE(
                backbone=self.hparams.PARE.BACKBONE,
                num_joints=self.hparams.PARE.NUM_JOINTS,
                softmax_temp=self.hparams.PARE.SOFTMAX_TEMP,
                num_features_smpl=self.hparams.PARE.NUM_FEATURES_SMPL,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_res=self.hparams.DATASET.IMG_RES,
                pretrained=self.hparams.TRAINING.PRETRAINED,
                iterative_regression=self.hparams.PARE.ITERATIVE_REGRESSION,
                num_iterations=self.hparams.PARE.NUM_ITERATIONS,
                iter_residual=self.hparams.PARE.ITER_RESIDUAL,
                shape_input_type=self.hparams.PARE.SHAPE_INPUT_TYPE,
                pose_input_type=self.hparams.PARE.POSE_INPUT_TYPE,
                pose_mlp_num_layers=self.hparams.PARE.POSE_MLP_NUM_LAYERS,
                shape_mlp_num_layers=self.hparams.PARE.SHAPE_MLP_NUM_LAYERS,
                pose_mlp_hidden_size=self.hparams.PARE.POSE_MLP_HIDDEN_SIZE,
                shape_mlp_hidden_size=self.hparams.PARE.SHAPE_MLP_HIDDEN_SIZE,
                use_keypoint_features_for_smpl_regression=self.hparams.PARE.
                USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
                use_heatmaps=self.hparams.DATASET.USE_HEATMAPS,
                use_keypoint_attention=self.hparams.PARE.
                USE_KEYPOINT_ATTENTION,
                use_postconv_keypoint_attention=self.hparams.PARE.
                USE_POSTCONV_KEYPOINT_ATTENTION,
                use_scale_keypoint_attention=self.hparams.PARE.
                USE_SCALE_KEYPOINT_ATTENTION,
                keypoint_attention_act=self.hparams.PARE.
                KEYPOINT_ATTENTION_ACT,
                use_final_nonlocal=self.hparams.PARE.USE_FINAL_NONLOCAL,
                use_branch_nonlocal=self.hparams.PARE.USE_BRANCH_NONLOCAL,
                use_hmr_regression=self.hparams.PARE.USE_HMR_REGRESSION,
                use_coattention=self.hparams.PARE.USE_COATTENTION,
                num_coattention_iter=self.hparams.PARE.NUM_COATTENTION_ITER,
                coattention_conv=self.hparams.PARE.COATTENTION_CONV,
                use_upsampling=self.hparams.PARE.USE_UPSAMPLING,
                deconv_conv_kernel_size=self.hparams.PARE.
                DECONV_CONV_KERNEL_SIZE,
                use_soft_attention=self.hparams.PARE.USE_SOFT_ATTENTION,
                num_branch_iteration=self.hparams.PARE.NUM_BRANCH_ITERATION,
                branch_deeper=self.hparams.PARE.BRANCH_DEEPER,
                num_deconv_layers=self.hparams.PARE.NUM_DECONV_LAYERS,
                num_deconv_filters=self.hparams.PARE.NUM_DECONV_FILTERS,
                use_resnet_conv_hrnet=self.hparams.PARE.USE_RESNET_CONV_HRNET,
                use_position_encodings=self.hparams.PARE.USE_POS_ENC,
                use_mean_camshape=self.hparams.PARE.USE_MEAN_CAMSHAPE,
                use_mean_pose=self.hparams.PARE.USE_MEAN_POSE,
                init_xavier=hparams.PARE.INIT_XAVIER,
            )
            self.loss_fn = PARELoss(
                shape_loss_weight=self.hparams.PARE.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.PARE.KEYPOINT_LOSS_WEIGHT,
                keypoint_native_loss_weight=self.hparams.PARE.
                KEYPOINT_NATIVE_LOSS_WEIGHT,
                heatmaps_loss_weight=self.hparams.PARE.HEATMAPS_LOSS_WEIGHT,
                part_loss_weight=self.hparams.PARE.PART_SEGM_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.PARE.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.PARE.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.PARE.OPENPOSE_TRAIN_WEIGHT,
                gt_train_weight=self.hparams.PARE.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.PARE.LOSS_WEIGHT,
                use_heatmaps_loss=self.hparams.DATASET.USE_HEATMAPS,
                smpl_part_loss_weight=self.hparams.PARE.SMPL_PART_LOSS_WEIGHT,
                use_shape_regularization=self.hparams.PARE.USE_SHAPE_REG,
            )
        # elif self.hparams.METHOD == 'nbf':
        #     from ..models import NBF
        #     from ..losses import PARELoss
        #     self.model = NBF(
        #         backbone=self.hparams.PARE.BACKBONE,
        #         num_joints=self.hparams.PARE.NUM_JOINTS,
        #         softmax_temp=self.hparams.PARE.SOFTMAX_TEMP,
        #         num_features_smpl=self.hparams.PARE.NUM_FEATURES_SMPL,
        #         focal_length=self.hparams.DATASET.FOCAL_LENGTH,
        #         img_res=self.hparams.DATASET.IMG_RES,
        #         pretrained=self.hparams.TRAINING.PRETRAINED,
        #         iterative_regression=self.hparams.PARE.ITERATIVE_REGRESSION,
        #         num_iterations=self.hparams.PARE.NUM_ITERATIONS,
        #         iter_residual=self.hparams.PARE.ITER_RESIDUAL,
        #         shape_input_type=self.hparams.PARE.SHAPE_INPUT_TYPE,
        #         pose_input_type=self.hparams.PARE.POSE_INPUT_TYPE,
        #         pose_mlp_num_layers=self.hparams.PARE.POSE_MLP_NUM_LAYERS,
        #         shape_mlp_num_layers=self.hparams.PARE.SHAPE_MLP_NUM_LAYERS,
        #         pose_mlp_hidden_size=self.hparams.PARE.POSE_MLP_HIDDEN_SIZE,
        #         shape_mlp_hidden_size=self.hparams.PARE.SHAPE_MLP_HIDDEN_SIZE,
        #         use_keypoint_features_for_smpl_regression=self.hparams.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
        #         use_heatmaps=self.hparams.DATASET.USE_HEATMAPS,
        #         use_keypoint_attention=self.hparams.PARE.USE_KEYPOINT_ATTENTION,
        #         use_postconv_keypoint_attention=self.hparams.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
        #         use_scale_keypoint_attention=self.hparams.PARE.USE_SCALE_KEYPOINT_ATTENTION,
        #         keypoint_attention_act=self.hparams.PARE.KEYPOINT_ATTENTION_ACT,
        #         use_final_nonlocal=self.hparams.PARE.USE_FINAL_NONLOCAL,
        #         use_branch_nonlocal=self.hparams.PARE.USE_BRANCH_NONLOCAL,
        #         use_hmr_regression=self.hparams.PARE.USE_HMR_REGRESSION,
        #         use_coattention=self.hparams.PARE.USE_COATTENTION,
        #         num_coattention_iter=self.hparams.PARE.NUM_COATTENTION_ITER,
        #         coattention_conv=self.hparams.PARE.COATTENTION_CONV,
        #         use_upsampling=self.hparams.PARE.USE_UPSAMPLING,
        #         deconv_conv_kernel_size=self.hparams.PARE.DECONV_CONV_KERNEL_SIZE,
        #         use_soft_attention=self.hparams.PARE.USE_SOFT_ATTENTION,
        #         num_branch_iteration=self.hparams.PARE.NUM_BRANCH_ITERATION,
        #         branch_deeper=self.hparams.PARE.BRANCH_DEEPER,
        #         num_deconv_layers=self.hparams.PARE.NUM_DECONV_LAYERS,
        #         num_deconv_filters=self.hparams.PARE.NUM_DECONV_FILTERS,
        #         use_resnet_conv_hrnet=self.hparams.PARE.USE_RESNET_CONV_HRNET,
        #         use_position_encodings=self.hparams.PARE.USE_POS_ENC,
        #         use_mean_camshape=self.hparams.PARE.USE_MEAN_CAMSHAPE,
        #         use_mean_pose=self.hparams.PARE.USE_MEAN_POSE,
        #         init_xavier=hparams.PARE.INIT_XAVIER,
        #     )
        #     self.loss_fn = PARELoss(
        #         shape_loss_weight=self.hparams.PARE.SHAPE_LOSS_WEIGHT,
        #         keypoint_loss_weight=self.hparams.PARE.KEYPOINT_LOSS_WEIGHT,
        #         keypoint_native_loss_weight=self.hparams.PARE.KEYPOINT_NATIVE_LOSS_WEIGHT,
        #         heatmaps_loss_weight=self.hparams.PARE.HEATMAPS_LOSS_WEIGHT,
        #         part_loss_weight=self.hparams.PARE.PART_SEGM_LOSS_WEIGHT,
        #         pose_loss_weight=self.hparams.PARE.POSE_LOSS_WEIGHT,
        #         beta_loss_weight=self.hparams.PARE.BETA_LOSS_WEIGHT,
        #         openpose_train_weight=self.hparams.PARE.OPENPOSE_TRAIN_WEIGHT,
        #         gt_train_weight=self.hparams.PARE.GT_TRAIN_WEIGHT,
        #         loss_weight=self.hparams.PARE.LOSS_WEIGHT,
        #         use_heatmaps_loss=self.hparams.DATASET.USE_HEATMAPS,
        #         smpl_part_loss_weight=self.hparams.PARE.SMPL_PART_LOSS_WEIGHT,
        #         use_shape_regularization=self.hparams.PARE.USE_SHAPE_REG,
        #     )
        # elif self.hparams.METHOD == 'spin':
        #     from ..models import HMR
        #     from ..losses import HMRLoss
        #     self.model = HMR(
        #         backbone=self.hparams.SPIN.BACKBONE,
        #         focal_length=self.hparams.DATASET.FOCAL_LENGTH,
        #         img_res=self.hparams.DATASET.IMG_RES,
        #         pretrained=self.hparams.TRAINING.PRETRAINED,
        #         p=self.hparams.TRAINING.DROPOUT_P,
        #         estimate_var=self.hparams.SPIN.ESTIMATE_UNCERTAINTY,
        #         uncertainty_activation=self.hparams.SPIN.UNCERTAINTY_ACTIVATION,
        #     )
        #     self.loss_fn = HMRLoss(
        #         shape_loss_weight=self.hparams.SPIN.SHAPE_LOSS_WEIGHT,
        #         keypoint_loss_weight=self.hparams.SPIN.KEYPOINT_LOSS_WEIGHT,
        #         pose_loss_weight=self.hparams.SPIN.POSE_LOSS_WEIGHT,
        #         beta_loss_weight=self.hparams.SPIN.BETA_LOSS_WEIGHT,
        #         openpose_train_weight=self.hparams.SPIN.OPENPOSE_TRAIN_WEIGHT,
        #         gt_train_weight=self.hparams.SPIN.GT_TRAIN_WEIGHT,
        #         loss_weight=self.hparams.SPIN.LOSS_WEIGHT,
        #         estimate_var=self.hparams.SPIN.ESTIMATE_UNCERTAINTY,
        #         uncertainty_loss=self.hparams.SPIN.UNCERTAINTY_LOSS,
        #         smpl_part_loss_weight=self.hparams.SPIN.SMPL_PART_LOSS_WEIGHT,
        #     )
        else:
            logger.error(f'{self.hparams.METHOD} is undefined!')
            exit()

        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.hparams.DATASET.BATCH_SIZE,
                         create_transl=False)
        self.add_module('smpl', self.smpl)

        # smpl_native regresses joint regressor with 24 smpl kinematic tree joints
        # It is used during training of PARE part branch to obtain 2d gt/predicted keypoints
        # in original SMPL coordinates
        self.smpl_native = SMPL_native(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False)
        self.add_module('smpl_native', self.smpl_native)

        render_resolution = self.hparams.DATASET.RENDER_RES if self.hparams.RUN_TEST \
            else self.hparams.DATASET.IMG_RES

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=render_resolution,
            faces=self.smpl.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

        if self.hparams.DATASET.USE_HEATMAPS in ('part_segm', 'part_segm_pool') or \
                self.hparams.TRAINING.USE_PART_SEGM_LOSS:
            # self.neural_renderer = nr.Renderer(
            #     dist_coeffs=None,
            #     orig_size=self.hparams.DATASET.IMG_RES,
            #     image_size=self.hparams.DATASET.IMG_RES,
            #     light_intensity_ambient=1,
            #     light_intensity_directional=0,
            #     anti_aliasing=False,
            # )

            self.register_buffer(
                'body_part_texture',
                get_body_part_texture(self.smpl.faces, n_vertices=6890))

            K, R = get_default_camera(
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                img_size=self.hparams.DATASET.IMG_RES)

            self.register_buffer('K', K)
            self.register_buffer('R', R)
            self.register_buffer(
                'smpl_faces',
                torch.from_numpy(self.smpl.faces.astype(
                    np.int32)).unsqueeze(0))
            # bins are discrete part labels, add 1 to avoid quantization error
            n_parts = 24
            self.register_buffer(
                'part_label_bins',
                (torch.arange(int(n_parts)) / float(n_parts) * 255.) + 1)

        # Initialize the training datasets only in training mode
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()

        self.example_input_array = torch.rand(1, 3,
                                              self.hparams.DATASET.IMG_RES,
                                              self.hparams.DATASET.IMG_RES)

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float())

        if len(self.val_ds) > 0:
            self.val_accuracy_results = {ds.dataset: [] for ds in self.val_ds}
        else:
            self.val_accuracy_results = []

        # Initialiatize variables required for evaluation
        self.init_evaluation_variables()

    def init_evaluation_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = []  # np.zeros(len(self.val_ds))
        self.val_pampjpe = []  # np.zeros(len(self.val_ds))
        self.val_v2v = []

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [],  # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [],  # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        # use this to save the errors for each image
        if self.hparams.TESTING.SAVE_IMAGES:
            self.val_images_errors = []

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam'] = []
            self.evaluation_results['vertices'] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # Get data from the batch
        images = batch['img']  # input image
        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas,
                           body_pose=gt_pose[:, 3:],
                           global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_size=self.hparams.DATASET.IMG_RES,
            use_all_joints=True
            if '3dpw' in self.hparams.DATASET.DATASETS_AND_RATIOS else False,
        )

        pred = self(images)

        camera_center = torch.zeros(batch_size, 2, device=self.device)

        batch['gt_cam_t'] = gt_cam_t
        batch['vertices'] = gt_vertices

        # generate heatmaps on the fly
        if self.hparams.DATASET.USE_HEATMAPS in ['hm', 'hm_soft', '']:
            #############################################
            # Obtain original 24 smpl gt 2d keypoints to supervise part branch of PARE
            # These keypoints are not used during vanilla HMR training
            gt_native_model_joints = self.smpl_native(
                betas=gt_betas,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3]).joints[:, :24, :]

            gt_smpl_keypoints_2d = perspective_projection(
                gt_native_model_joints,
                rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(
                    batch_size, -1, -1),
                translation=gt_cam_t,
                focal_length=self.hparams.DATASET.FOCAL_LENGTH,
                camera_center=camera_center,
            )
            # Normalize keypoints to [-1,1]
            gt_smpl_keypoints_2d = gt_smpl_keypoints_2d / (
                self.hparams.DATASET.IMG_RES / 2.)
            batch['smpl_keypoints'] = gt_smpl_keypoints_2d
            #############################################

            batch['heatmaps_2d'], batch[
                'joint_vis'] = generate_heatmaps_2d_batch(
                    joints=gt_smpl_keypoints_2d,
                    num_joints=self.hparams.PARE.NUM_JOINTS,
                    heatmap_size=self.hparams.DATASET.IMG_RES // 4,
                    image_size=self.hparams.DATASET.IMG_RES,
                    sigma=(2. / 64.) * self.hparams.DATASET.IMG_RES // 4,
                )

        if self.hparams.DATASET.USE_HEATMAPS in [
                'part_segm', 'part_segm_pool'
        ] or self.hparams.TRAINING.USE_PART_SEGM_LOSS:
            batch['gt_segm_mask'], batch['gt_segm_rgb'] = generate_part_labels(
                vertices=gt_vertices,
                faces=self.smpl_faces,
                cam_t=gt_cam_t,
                K=self.K,
                R=self.R,
                body_part_texture=self.body_part_texture,
                # neural_renderer=self.neural_renderer,
                part_bins=self.part_label_bins,
            )

        if self.hparams.TRAINING.USE_PART_SEGM_LOSS:
            _, pred['pred_segm_rgb'] = generate_part_labels(
                vertices=pred['smpl_vertices'],
                faces=self.smpl_faces,
                cam_t=pred['pred_cam_t'],
                K=self.K,
                R=self.R,
                body_part_texture=self.body_part_texture,
                # neural_renderer=self.neural_renderer,
                part_bins=self.part_label_bins,
            )

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        tensorboard_logs = loss_dict

        self.log_dict(tensorboard_logs)

        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0:
            self.train_summaries(input_batch=batch, output=pred)

        return {'loss': loss, 'log': tensorboard_logs}

    def train_summaries(self, input_batch, output):
        images = input_batch['img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()
        opt_vertices = input_batch['vertices']

        pred_cam_t = output['pred_cam_t'].detach()
        opt_cam_t = input_batch['gt_cam_t']

        pred_kp_2d = output['pred_kp2d'].detach(
        ) if 'pred_kp2d' in output.keys() else None
        gt_kp_2d = input_batch[
            'smpl_keypoints'] if 'smpl_keypoints' in input_batch.keys(
            ) else None

        vertex_colors = None

        if 'pred_pose_var' in output.keys():
            # color the vertices with uncertainty estimations
            from ..utils.vis_utils import color_vertices_batch
            per_joint_label = output['pred_pose_var'][:, 144:].reshape(
                -1, 24, 6).mean(-1).detach().cpu().numpy()
            vertex_colors = color_vertices_batch(per_joint_label)

        if 'pred_heatmaps_2d' in output.keys():
            pred_heatmaps = output['pred_heatmaps_2d'].detach().cpu().numpy()
            gt_heatmaps = input_batch['heatmaps_2d'].detach().cpu().numpy()

        images_pred = self.renderer.visualize_tb(
            pred_vertices,
            pred_cam_t,
            images,
            kp_2d=pred_kp_2d,
            sideview=self.hparams.TESTING.SIDEVIEW,
            vertex_colors=vertex_colors,
        )

        images_opt = self.renderer.visualize_tb(
            opt_vertices,
            opt_cam_t,
            images,
            kp_2d=gt_kp_2d,
            sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.hparams.TRAINING.SAVE_IMAGES:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

            images_opt = images_opt.cpu().numpy().transpose(1, 2, 0) * 255
            images_opt = np.clip(images_opt, 0, 255).astype(np.uint8)

            save_dir = os.path.join(self.hparams.LOG_DIR, 'training_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_dir, f'result_{self.global_step:08d}.jpg'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB))
            cv2.imwrite(
                os.path.join(save_dir, f'gt_{self.global_step:08d}.jpg'),
                cv2.cvtColor(images_opt, cv2.COLOR_BGR2RGB))

    def validation_step(self,
                        batch,
                        batch_nb,
                        dataloader_nb,
                        vis=False,
                        save=True,
                        mesh_save_dir=None):
        images = batch['img']
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred = self(images)
            pred_vertices = pred['smpl_vertices']

        joint_mapper_h36m = constants.H36M_TO_J17 if dataset_names[0] == 'mpi-inf-3dhp' \
            else constants.H36M_TO_J14

        if dataset_names[0] == 'mpii':
            # Only for qualitative result experiments
            if self.hparams.TESTING.SAVE_IMAGES:
                self.validation_summaries(batch, pred, batch_nb, dataloader_nb)
                error, r_error = torch.zeros(1), torch.zeros(1)
                error_per_joint, r_error_per_joint = torch.zeros(
                    14), torch.zeros(14)
            else:
                logger.error(
                    'Set `TESTING.SAVE_IMAGES` to `True` when using ITW datasets the evaluation dataset'
                )
                exit()
        else:
            J_regressor_batch = self.J_regressor[None, :].expand(
                pred_vertices.shape[0], -1, -1)

            gt_keypoints_3d = batch['pose_3d'].cuda()
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(
                ((pred_keypoints_3d -
                  gt_keypoints_3d)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            idx_start = batch_nb * self.hparams.DATASET.BATCH_SIZE
            idx_stop = batch_nb * self.hparams.DATASET.BATCH_SIZE + curr_batch_size

            # Reconstuction_error
            r_error, r_error_per_joint = reconstruction_error(
                pred_keypoints_3d.cpu().numpy(),
                gt_keypoints_3d.cpu().numpy(),
                reduction=None,
            )

            # Per-vertex error
            if 'vertices' in batch.keys():
                gt_vertices = batch['vertices'].cuda()

                v2v = compute_error_verts(
                    pred_verts=pred_vertices.cpu().numpy(),
                    target_verts=gt_vertices.cpu().numpy(),
                )
                self.val_v2v += v2v.tolist()
            else:
                self.val_v2v += np.zeros_like(error).tolist()

            ####### DEBUG 3D JOINT PREDICTIONS and GT ###########
            # from ..utils.vis_utils import show_3d_pose
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(12, 7))
            # plt.title(f'error {error[0].item()*1000:.2f}, r_err {r_error[0].item()*1000:.2f}')
            # ax = fig.add_subplot('121', projection='3d', aspect='auto')
            # show_3d_pose(kp_3d=pred_keypoints_3d[0].cpu(), ax=ax)
            #
            # ax = fig.add_subplot('122', projection='3d', aspect='auto')
            # show_3d_pose(kp_3d=gt_keypoints_3d[0].cpu(), ax=ax)
            # plt.show()
            #####################################################

            self.val_mpjpe += error.tolist()
            self.val_pampjpe += r_error.tolist()

            error_per_joint = torch.sqrt(
                ((pred_keypoints_3d -
                  gt_keypoints_3d)**2).sum(dim=-1)).cpu().numpy()

            self.evaluation_results['mpjpe'] += error_per_joint[:, :14].tolist(
            )
            self.evaluation_results[
                'pampjpe'] += r_error_per_joint[:, :14].tolist()

            if 'vertices' in batch.keys():
                self.evaluation_results['v2v'] += v2v.tolist()
            else:
                self.evaluation_results['v2v'] += np.zeros_like(error).tolist()

            self.evaluation_results['imgname'] += imgnames
            self.evaluation_results['dataset_name'] += dataset_names

            if self.hparams.TESTING.SAVE_RESULTS:
                tolist = lambda x: [i for i in x.cpu().numpy()]
                self.evaluation_results['pose'] += tolist(pred['pred_pose'])
                self.evaluation_results['shape'] += tolist(pred['pred_shape'])
                self.evaluation_results['cam'] += tolist(pred['pred_cam'])
                self.evaluation_results['vertices'] += tolist(pred_vertices)

            if vis:
                # this doesn't save the resulting images
                vis_image = self.validation_summaries(
                    batch,
                    pred,
                    batch_nb,
                    dataloader_nb,
                    save=save,
                    error=error,
                    r_error=r_error,
                    per_joint_error=error_per_joint,
                    mesh_save_dir=mesh_save_dir,
                )
                return {
                    'pred_keypoints_3d':
                    pred_keypoints_3d,
                    'gt_keypoints_3d':
                    gt_keypoints_3d,
                    'pred_kp2d':
                    pred['pred_kp2d'] if 'pred_kp2d' in pred.keys() else None,
                    'mpjpe':
                    error.mean(),
                    'pampjpe':
                    r_error.mean(),
                    'per_mpjpe':
                    error_per_joint,
                    'per_pampjpe':
                    r_error_per_joint,
                    'vis_image':
                    vis_image,
                }

            if self.hparams.TESTING.SAVE_IMAGES:
                # this saves the rendered images
                self.validation_summaries(batch,
                                          pred,
                                          batch_nb,
                                          dataloader_nb,
                                          save=save,
                                          error=error,
                                          r_error=r_error,
                                          per_joint_error=error_per_joint)

        return {
            'mpjpe': error.mean(),
            'pampjpe': r_error.mean(),
            'per_mpjpe': error_per_joint,
            'per_pampjpe': r_error_per_joint
        }

    def validation_summaries(self,
                             input_batch,
                             output,
                             batch_idx,
                             dataloader_nb,
                             save=True,
                             error=None,
                             r_error=None,
                             per_joint_error=None,
                             kp_3d=None,
                             mesh_save_dir=None):
        # images = input_batch['img']
        images = input_batch['disp_img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()
        # pred_cam_t = output['pred_cam_t'].detach()
        pred_kp_2d = output['pred_kp2d'].detach(
        ) if 'pred_kp2d' in output.keys() else None

        ########### convert camera parameters to display image params ###########
        pred_cam = output['pred_cam'].detach()
        pred_cam_t = convert_weak_perspective_to_perspective(
            pred_cam,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.RENDER_RES,
        )

        # joint_labels = None
        vertex_colors = None
        per_joint_label = None
        pred_heatmaps = None

        if 'pred_pose_var' in output.keys():
            # color the vertices with uncertainty estimations
            per_joint_label = output['pred_pose_var'][:, 144:].reshape(
                -1, 24, 6).mean(-1).detach().cpu().numpy()
            # per_joint_label = per_joint_label ** 2
            vertex_colors = color_vertices_batch(per_joint_label)

        if 'pred_heatmaps_2d' in output.keys():
            pred_heatmaps = output['pred_heatmaps_2d'].detach().cpu().numpy()

        mesh_filename = None
        if self.hparams.TESTING.SAVE_MESHES:
            save_dir = mesh_save_dir if mesh_save_dir else os.path.join(
                self.hparams.LOG_DIR, 'output_meshes')
            os.makedirs(save_dir, exist_ok=True)
            mesh_filename = os.path.join(
                save_dir, f'result_{dataloader_nb:02d}_{batch_idx:05d}.obj')

            images_mesh = images[0].clone().cpu().numpy().transpose(1, 2,
                                                                    0) * 255
            images_mesh = np.clip(images_mesh, 0, 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    save_dir,
                    f'result_{dataloader_nb:02d}_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_mesh, cv2.COLOR_BGR2RGB))

        images_pred = self.renderer.visualize_tb(
            pred_vertices,
            pred_cam_t,
            images,
            pred_kp_2d,
            nb_max_img=1,
            sideview=self.hparams.TESTING.SIDEVIEW,
            vertex_colors=vertex_colors,
            joint_labels=per_joint_error *
            1000. if per_joint_error is not None else None,
            joint_uncertainty=per_joint_label
            if per_joint_label is not None else None,
            heatmaps=pred_heatmaps,
            multi_sideview=self.hparams.TESTING.MULTI_SIDEVIEW,
            mesh_filename=mesh_filename,
        )

        # self.logger.experiment.add_image('pred_shape', images_pred, self.global_step)
        images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
        images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)

        if error is not None and r_error is not None and self.hparams.TESTING.SAVE_IMAGES:
            # log the errors of saved images
            self.val_images_errors.append([error[0], r_error[0]])
            # draw the errors as text on saved images
            images_pred = cv2.putText(
                images_pred,
                f'{batch_idx}, e: {error[0] * 1000:.1f}, re: {r_error[0] * 1000:.1f}',
                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        if save:
            save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(
                    save_dir,
                    f'result_{dataloader_nb:02d}_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB))
        return cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)

    def validation_epoch_end(self, outputs):
        if 'coco' in self.val_ds or 'mpii' in self.val_ds:
            logger.info('...THE END...')
            exit()

        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_v2v = np.array(self.val_v2v)

        for k, v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        if len(self.val_ds) == 1:
            avg_mpjpe, avg_pampjpe = 1000 * self.val_mpjpe.mean(
            ), 1000 * self.val_pampjpe.mean()
            avg_v2v = 1000 * self.val_v2v.mean()

            logger.info(f'***** Epoch {self.current_epoch} *****')
            logger.info('MPJPE: ' + str(avg_mpjpe))
            logger.info('PA-MPJPE: ' + str(avg_pampjpe))
            logger.info('V2V (mm): ' + str(avg_v2v))

            acc = {
                'val_mpjpe': avg_mpjpe.item(),
                'val_pampjpe': avg_pampjpe.item(),
                'val_v2v': avg_v2v.item(),
            }

            self.val_save_best_results(acc)

            # save the mpjpe and pa-mpjpe results per image
            if self.hparams.TESTING.SAVE_IMAGES and len(
                    self.val_images_errors) > 0:
                save_path = os.path.join(self.hparams.LOG_DIR,
                                         'val_images_error.npy')
                logger.info(f'Saving the errors of images {save_path}')
                np.save(save_path, np.asarray(self.val_images_errors))

            # save the detailed experiment results for post-analysis script
            joblib.dump(
                self.evaluation_results,
                os.path.join(
                    self.hparams.LOG_DIR,
                    f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl'))

            avg_mpjpe, avg_pampjpe = torch.tensor(avg_mpjpe), torch.tensor(
                avg_pampjpe)
            tensorboard_logs = {
                'val/val_mpjpe': avg_mpjpe,
                'val/val_pampjpe': avg_pampjpe,
            }
            val_log = {
                'val_loss': avg_pampjpe,
                'val_mpjpe': avg_mpjpe,
                'val_pampjpe': avg_pampjpe,
                'log': tensorboard_logs
            }
        else:
            logger.info(f'***** Epoch {self.current_epoch} *****')
            val_log = {}
            val_log['log'] = {}

            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                idxs = self.evaluation_results['dataset_name'] == ds_name

                mpjpe = 1000 * self.val_mpjpe[idxs].mean()
                pampjpe = 1000 * self.val_pampjpe[idxs].mean()
                v2v = 1000 * self.val_v2v[idxs].mean()

                logger.info(f'{ds_name} MPJPE: ' + str(mpjpe))
                logger.info(f'{ds_name} PA-MPJPE: ' + str(pampjpe))
                logger.info(f'{ds_name} V2V: ' + str(v2v))

                acc = {
                    'val_mpjpe': mpjpe.item(),
                    'val_pampjpe': pampjpe.item(),
                    'val_v2v': v2v.item(),
                }

                val_log[f'val_mpjpe_{ds_name}'] = mpjpe
                val_log[f'val_pampjpe_{ds_name}'] = pampjpe

                val_log['log'][f'val/val_mpjpe_{ds_name}'] = mpjpe
                val_log['log'][f'val/val_pampjpe_{ds_name}'] = pampjpe

                self.val_save_best_results(acc, ds_name)

                # save the mpjpe and pa-mpjpe results per image
                if self.hparams.TESTING.SAVE_IMAGES and len(
                        self.val_images_errors) > 0:
                    save_path = os.path.join(self.hparams.LOG_DIR,
                                             'val_images_error.npy')
                    logger.info(f'Saving the errors of images {save_path}')
                    np.save(save_path, np.asarray(self.val_images_errors))

                eval_res = {
                    k: v[idxs]
                    for k, v in self.evaluation_results.items()
                }
                joblib.dump(
                    eval_res,
                    os.path.join(self.hparams.LOG_DIR,
                                 f'evaluation_results_{ds_name}.pkl'))

                # always set the first dataset as the main one
                if ds_idx == 0:
                    avg_mpjpe, avg_pampjpe = mpjpe, pampjpe
                    val_log['val_loss'] = avg_pampjpe
                    val_log['val_mpjpe'] = avg_mpjpe
                    val_log['val_pampjpe'] = avg_pampjpe

                    val_log['log'][f'val/val_mpjpe'] = avg_mpjpe
                    val_log['log'][f'val/val_pampjpe'] = avg_pampjpe

        for k, v in val_log.items():
            if k == 'log':
                pass
            else:
                self.log(k, v)

        # reset evaluation variables
        self.init_evaluation_variables()
        return val_log

    def test_step(self, batch, batch_nb, dataloader_nb):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.OPTIMIZER.LR,
                                weight_decay=self.hparams.OPTIMIZER.WD)

    def train_dataset(self):
        if self.hparams.DATASET.TRAIN_DS == 'all':
            train_ds = EFTDataset(options=self.hparams.DATASET,
                                  ignore_3d=self.hparams.DATASET.IGNORE_3D,
                                  is_train=True)
        elif self.hparams.DATASET.TRAIN_DS == 'stage':
            # stage dataset is used to
            stage_datasets = self.hparams.DATASET.STAGE_DATASETS.split(' ')
            stage_dict = {
                x.split('+')[0]: x.split('+')[1]
                for x in stage_datasets
            }
            assert self.hparams.DATASET.STAGE_DATASETS.startswith(
                '0'), 'Stage datasets should start from epoch 0'

            if str(self.current_epoch) in stage_dict.keys():
                self.hparams.DATASET.DATASETS_AND_RATIOS = stage_dict[str(
                    self.current_epoch)]

            train_ds = EFTDataset(options=self.hparams.DATASET,
                                  ignore_3d=self.hparams.DATASET.IGNORE_3D,
                                  is_train=True)
        else:
            train_ds = eval(f'{self.hparams.DATASET.LOAD_TYPE}Dataset')(
                options=self.hparams.DATASET,
                dataset=self.hparams.DATASET.TRAIN_DS,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                num_images=self.hparams.DATASET.TRAIN_NUM_IMAGES,
                is_train=True,
            )

        return train_ds

    def train_dataloader(self):
        set_seed(self.hparams.SEED_VALUE)

        self.train_ds = self.train_dataset()

        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )
        # return CheckpointDataLoader(
        #     dataset=self.train_ds,
        #     batch_size=self.hparams.DATASET.BATCH_SIZE,
        #     num_workers=self.hparams.DATASET.NUM_WORKERS,
        #     pin_memory=self.hparams.DATASET.PIN_MEMORY,
        #     shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        # )

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        num_images = [self.hparams.DATASET.TEST_NUM_IMAGES] * len(datasets)
        # num_images[0] = self.hparams.DATASET.TEST_NUM_IMAGES
        for idx, dataset_name in enumerate(datasets):
            val_datasets.append(
                eval(f'{self.hparams.DATASET.LOAD_TYPE}Dataset')(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    num_images=num_images[idx],
                    is_train=False,
                ))

        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                ))
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()

    def val_save_best_results(self, acc, ds_name=None):
        # log the running training metrics
        if ds_name:
            fname = f'val_accuracy_results_{ds_name}.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results[ds_name].append(
                [self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results[ds_name], f, indent=4)
        else:
            fname = 'val_accuracy_results.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results.append(
                [self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results, f, indent=4)
