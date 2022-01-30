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
import time
import torch
import joblib
import colorsys
import numpy as np
from tqdm import tqdm
from loguru import logger
from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from ..models import PARE, HMR
from .config import update_hparams
from ..utils.vibe_renderer import Renderer
from ..utils.pose_tracker import run_posetracker
from ..utils.train_utils import load_pretrained_model
from ..dataset.inference import Inference, ImageFolder
from ..utils.smooth_pose import smooth_pose
from ..utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
)
from ..utils.vibe_image_utils import get_single_image_crop_demo

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

    def run_tracking(self, video_file, image_folder):
        # ========= Run tracking ========= #
        if self.args.tracking_method == 'pose':
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file,
                                               staf_folder=self.args.staf_dir,
                                               display=self.args.display)
        else:
            # run multi object tracker
            mot = MPT(
                device=self.device,
                batch_size=self.args.tracker_batch_size,
                display=self.args.display,
                detector_type=self.args.detector,
                output_format='dict',
                yolo_img_size=self.args.yolo_img_size,
            )
            tracking_results = mot(image_folder)

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        return tracking_results

    def run_detector(self, image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=self.args.display,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = mot.detect(image_folder)
        return bboxes

    @torch.no_grad()
    def run_on_image_folder(self,
                            image_folder,
                            detections,
                            output_path,
                            output_img_folder,
                            bbox_scale=1.0):
        image_file_names = [
            os.path.join(image_folder, x) for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        image_file_names = sorted(image_file_names)

        for img_idx, img_fname in enumerate(image_file_names):
            dets = detections[img_idx]

            if len(dets) < 1:
                continue

            img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            orig_height, orig_width = img.shape[:2]

            inp_images = torch.zeros(len(dets),
                                     3,
                                     self.model_cfg.DATASET.IMG_RES,
                                     self.model_cfg.DATASET.IMG_RES,
                                     device=self.device,
                                     dtype=torch.float)

            for det_idx, det in enumerate(dets):
                bbox = det
                norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                    img,
                    bbox,
                    kp_2d=None,
                    scale=bbox_scale,
                    crop_size=self.model_cfg.DATASET.IMG_RES)
                inp_images[det_idx] = norm_img.float().to(self.device)
            try:
                output = self.model(inp_images)
            except Exception as e:
                import IPython
                IPython.embed()
                exit()

            for k, v in output.items():
                output[k] = v.cpu().numpy()

            orig_cam = convert_crop_cam_to_orig_img(cam=output['pred_cam'],
                                                    bbox=dets,
                                                    img_width=orig_width,
                                                    img_height=orig_height)

            smpl_joints2d = convert_crop_coords_to_orig_img(
                bbox=dets,
                keypoints=output['smpl_joints2d'],
                crop_size=self.model_cfg.DATASET.IMG_RES,
            )

            output['bboxes'] = dets
            output['orig_cam'] = orig_cam
            output['smpl_joints2d'] = smpl_joints2d

            del inp_images

            if not self.args.no_save:
                save_f = os.path.join(
                    output_path, 'pare_results',
                    os.path.basename(img_fname).replace(
                        img_fname.split('.')[-1], 'pkl'))
                joblib.dump(output, save_f)

            if not self.args.no_render:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                renderer = Renderer(resolution=(orig_width, orig_height),
                                    orig_img=True,
                                    wireframe=self.args.wireframe)

                if self.args.sideview:
                    side_img = np.zeros_like(img)

                for idx in range(len(dets)):
                    verts = output['smpl_vertices'][idx]
                    cam = output['orig_cam'][idx]
                    keypoints = output['smpl_joints2d'][idx]

                    mc = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

                    mesh_filename = None

                    if self.args.save_obj:
                        mesh_folder = os.path.join(
                            output_path, 'meshes',
                            os.path.basename(img_fname).split('.')[0])
                        os.makedirs(mesh_folder, exist_ok=True)
                        mesh_filename = os.path.join(mesh_folder,
                                                     f'{idx:06d}.obj')

                    img = renderer.render(
                        img,
                        verts,
                        cam=cam,
                        color=mc,
                        mesh_filename=mesh_filename,
                    )

                    if self.args.draw_keypoints:
                        for idx, pt in enumerate(keypoints):
                            cv2.circle(img, (pt[0], pt[1]), 4, (0, 255, 0), -1)

                    if self.args.sideview:
                        side_img = renderer.render(
                            side_img,
                            verts,
                            cam=cam,
                            color=mc,
                            angle=270,
                            axis=[0, 1, 0],
                        )

                if self.args.sideview:
                    img = np.concatenate([img, side_img], axis=1)

                cv2.imwrite(
                    os.path.join(output_img_folder,
                                 os.path.basename(img_fname)), img)

                if self.args.display:
                    cv2.imshow('Video', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if self.args.display:
            cv2.destroyAllWindows()

    @torch.no_grad()
    def run_on_video(self,
                     tracking_results,
                     image_folder,
                     orig_width,
                     orig_height,
                     bbox_scale=1.0):
        # ========= Run PARE on each person ========= #
        logger.info(f'Running PARE on each tracklet...')

        pare_results = {}
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            if self.args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif self.args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset,
                                    batch_size=self.args.batch_size,
                                    num_workers=8)

            pred_cam, pred_verts, pred_pose, pred_betas, \
            pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                output = self.model(batch)

                pred_cam.append(
                    output['pred_cam'])  # [:, :, :3].reshape(batch_size, -1))
                pred_verts.append(output['smpl_vertices']
                                  )  # .reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(
                    output['pred_pose']
                )  # [:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(
                    output['pred_shape']
                )  # [:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['smpl_joints3d']
                                     )  # .reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['smpl_joints2d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            del batch

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            if self.args.smooth:
                min_cutoff = self.args.min_cutoff  # 0.004
                beta = self.args.beta  # 1.5
                logger.info(
                    f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}'
                )
                pred_verts, pred_pose, pred_joints3d = smooth_pose(
                    pred_pose, pred_betas, min_cutoff=min_cutoff, beta=beta)

            orig_cam = convert_crop_cam_to_orig_img(cam=pred_cam,
                                                    bbox=bboxes,
                                                    img_width=orig_width,
                                                    img_height=orig_height)
            logger.info(
                'Converting smpl keypoints 2d to original image coordinate')

            smpl_joints2d = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=self.model_cfg.DATASET.IMG_RES,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'smpl_joints2d': smpl_joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            pare_results[person_id] = output_dict
        return pare_results

    def render_results(self, pare_results, image_folder, output_img_folder,
                       output_path, orig_width, orig_height, num_frames):
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height),
                            orig_img=True,
                            wireframe=self.args.wireframe)

        logger.info(
            f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(pare_results, num_frames)
        mesh_color = {
            k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
            for k in pare_results.keys()
        }

        image_file_names = sorted([
            os.path.join(image_folder, x) for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if self.args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                frame_kp = person_data['joints2d']

                mc = mesh_color[person_id]

                mesh_filename = None

                if self.args.save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes',
                                               f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder,
                                                 f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if self.args.draw_keypoints:
                    for idx, pt in enumerate(frame_kp):
                        cv2.circle(img, (pt[0], pt[1]), 4, (0, 255, 0), -1)

                if self.args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0, 1, 0],
                    )

            if self.args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(
                os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if self.args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.args.display:
            cv2.destroyAllWindows()
