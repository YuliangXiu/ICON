# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at pixie@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
import cv2

from .models.encoders import ResnetEncoder, MLP, HRNEncoder
from .models.moderators import TempSoftmaxFusion
from .models.SMPLX import SMPLX
from .utils import util
from .utils import rotation_converter as converter
from .utils import tensor_cropper
from .utils.config import cfg


class PIXIE(object):
    def __init__(self, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        # parameters setting
        self.param_list_dict = {}
        for lst in self.cfg.params.keys():
            param_list = cfg.params.get(lst)
            self.param_list_dict[lst] = {
                i: cfg.model.get('n_'+i) for i in param_list}

        # Build the models
        self._create_model()
        # Set up the cropping modules used to generate face/hand crops from the body predictions
        self._setup_cropper()

    def forward(self, data):

        # encode + decode
        param_dict = self.encode(
            {'body': {'image': data}}, threthold=True, keep_local=True, copy_and_paste=False)
        opdict = self.decode(param_dict['body'], param_type='body')

        return opdict

    def _setup_cropper(self):
        self.Cropper = {}
        for crop_part in ['head', 'hand']:
            data_cfg = self.cfg.dataset[crop_part]
            scale_size = (data_cfg.scale_min + data_cfg.scale_max)*0.5
            self.Cropper[crop_part] = tensor_cropper.Cropper(
                crop_size=data_cfg.image_size,
                scale=[scale_size, scale_size],
                trans_scale=0)

    def _create_model(self):
        self.model_dict = {}
        # Build all image encoders
        # Hand encoder only works for right hand, for left hand, flip inputs and flip the results back
        self.Encoder = {}
        for key in self.cfg.network.encoder.keys():
            if self.cfg.network.encoder.get(key).type == 'resnet50':
                self.Encoder[key] = ResnetEncoder().to(self.device)
            elif self.cfg.network.encoder.get(key).type == 'hrnet':
                self.Encoder[key] = HRNEncoder().to(self.device)
            self.model_dict[f'Encoder_{key}'] = self.Encoder[key].state_dict()

        # Build the parameter regressors
        self.Regressor = {}
        for key in self.cfg.network.regressor.keys():
            n_output = sum(self.param_list_dict[f'{key}_list'].values())
            channels = [2048] + \
                self.cfg.network.regressor.get(key).channels + [n_output]
            if self.cfg.network.regressor.get(key).type == 'mlp':
                self.Regressor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f'Regressor_{key}'] = self.Regressor[key].state_dict(
            )

        # Build the extractors
        # to extract separate head/left hand/right hand feature from body feature
        self.Extractor = {}
        for key in self.cfg.network.extractor.keys():
            channels = [2048] + \
                self.cfg.network.extractor.get(key).channels + [2048]
            if self.cfg.network.extractor.get(key).type == 'mlp':
                self.Extractor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f'Extractor_{key}'] = self.Extractor[key].state_dict(
            )

        # Build the moderators
        self.Moderator = {}
        for key in self.cfg.network.moderator.keys():
            share_part = key.split('_')[0]
            detach_inputs = self.cfg.network.moderator.get(key).detach_inputs
            detach_feature = self.cfg.network.moderator.get(key).detach_feature
            channels = [2048*2] + \
                self.cfg.network.moderator.get(key).channels + [2]
            self.Moderator[key] = TempSoftmaxFusion(
                detach_inputs=detach_inputs, detach_feature=detach_feature,
                channels=channels).to(self.device)
            self.model_dict[f'Moderator_{key}'] = self.Moderator[key].state_dict(
            )

        # Build the SMPL-X body model, which we also use to represent faces and
        # hands, using the relevant parts only
        self.smplx = SMPLX(self.cfg.model).to(self.device)
        self.part_indices = self.smplx.part_indices

        # -- resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            for key in self.model_dict.keys():
                util.copy_state_dict(self.model_dict[key], checkpoint[key])
        else:
            print(f'pixie trained model path: {model_path} does not exist!')
            exit()
        # eval mode
        for module in [self.Encoder, self.Regressor, self.Moderator, self.Extractor]:
            for net in module.values():
                net.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
        return code_dict

    def part_from_body(self, image, part_key, points_dict, crop_joints=None):
        ''' crop part(head/left_hand/right_hand) out from body data, joints also change accordingly
        '''
        assert part_key in ['head', 'left_hand', 'right_hand']
        assert 'smplx_kpt' in points_dict.keys()
        if part_key == 'head':
            # use face 68 kpts for cropping head image
            indices_key = 'face'
        elif part_key == 'left_hand':
            indices_key = 'left_hand'
        elif part_key == 'right_hand':
            indices_key = 'right_hand'

        # get points for cropping
        part_indices = self.part_indices[indices_key]
        if crop_joints is not None:
            points_for_crop = crop_joints[:, part_indices]
        else:
            points_for_crop = points_dict['smplx_kpt'][:, part_indices]

        # crop
        cropper_key = 'hand' if 'hand' in part_key else part_key
        points_scale = image.shape[-2:]
        cropped_image, tform = self.Cropper[cropper_key].crop(
            image,
            points_for_crop,
            points_scale
        )
        # transform points(must be normalized to [-1.1]) accordingly
        cropped_points_dict = {}
        for points_key in points_dict.keys():
            points = points_dict[points_key]
            cropped_points = self.Cropper[cropper_key].transform_points(
                points, tform, points_scale, normalize=True)
            cropped_points_dict[points_key] = cropped_points
        return cropped_image, cropped_points_dict

    @torch.no_grad()
    def encode(self, data, threthold=True, keep_local=True, copy_and_paste=False, body_only=False):
        ''' Encode images to smplx parameters
        Args:
            data: dict
                key: image_type (body/head/hand)
                value: 
                    image: [bz, 3, 224, 224], range [0,1]
                    image_hd(needed if key==body): a high res version of image, only for cropping parts from body image
                    head_image: optinal, well-cropped head from body image
                    left_hand_image: optinal, well-cropped left hand from body image
                    right_hand_image: optinal, well-cropped right hand from body image
        Returns:
            param_dict: dict
                key: image_type (body/head/hand)
                value: param_dict
        '''
        for key in data.keys():
            assert key in ['body', 'head', 'hand']

        feature = {}
        param_dict = {}

        # Encode features
        for key in data.keys():
            part = key
            # encode feature
            feature[key] = {}
            feature[key][part] = self.Encoder[part](data[key]['image'])

            # for head/hand image
            if key == 'head' or key == 'hand':
                # predict head/hand-only parameters from part feature
                part_dict = self.decompose_code(self.Regressor[part](
                    feature[key][part]), self.param_list_dict[f'{part}_list'])
                # if input is part data, skip feature fusion: share feature is the same as part feature
                # then predict share parameters
                feature[key][f'{key}_share'] = feature[key][key]
                share_dict = self.decompose_code(
                    self.Regressor[f'{part}_share'](
                        feature[key][f'{part}_share']),
                    self.param_list_dict[f'{part}_share_list'])
                # compose parameters
                param_dict[key] = {**share_dict, **part_dict}

            # for body image
            if key == 'body':
                fusion_weight = {}
                f_body = feature['body']['body']
                # extract part feature
                for part_name in ['head', 'left_hand', 'right_hand']:
                    feature['body'][f'{part_name}_share'] = self.Extractor[f'{part_name}_share'](
                        f_body)

                # -- check if part crops are given, if not, crop parts by coarse body estimation
                if 'head_image' not in data[key].keys() \
                        or 'left_hand_image' not in data[key].keys() \
                        or 'right_hand_image' not in data[key].keys():
                    # - run without fusion to get coarse estimation, for cropping parts
                    # body only
                    body_dict = self.decompose_code(self.Regressor[part](
                        feature[key][part]), self.param_list_dict[part+'_list'])
                    # head share
                    head_share_dict = self.decompose_code(self.Regressor['head'+'_share'](
                        feature[key]['head'+'_share']), self.param_list_dict['head'+'_share_list'])
                    # right hand share
                    right_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                        feature[key]['right_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                    # left hand share
                    left_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                        feature[key]['left_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                    # change the dict name from right to left
                    left_hand_share_dict['left_hand_pose'] = left_hand_share_dict.pop(
                        'right_hand_pose')
                    left_hand_share_dict['left_wrist_pose'] = left_hand_share_dict.pop(
                        'right_wrist_pose')
                    param_dict[key] = {**body_dict, **head_share_dict,
                                       **left_hand_share_dict, **right_hand_share_dict}
                    if body_only:
                        param_dict['moderator_weight'] = None
                        return param_dict
                    prediction_body_only = self.decode(
                        param_dict[key], param_type='body')
                    # crop
                    for part_name in ['head', 'left_hand', 'right_hand']:
                        part = part_name.split('_')[-1]
                        points_dict = {
                            'smplx_kpt': prediction_body_only['smplx_kpt'],
                            'trans_verts': prediction_body_only['transformed_vertices']
                        }
                        image_hd = torchvision.transforms.Resize(
                            1024)(data['body']['image'])
                        cropped_image, cropped_joints_dict = self.part_from_body(
                            image_hd, part_name, points_dict)
                        data[key][part_name+'_image'] = cropped_image

                # -- encode features from part crops, then fuse feature using the weight from moderator
                for part_name in ['head', 'left_hand', 'right_hand']:
                    part = part_name.split('_')[-1]
                    cropped_image = data[key][part_name+'_image']
                    # if left hand, flip it as if it is right hand
                    if part_name == 'left_hand':
                        cropped_image = torch.flip(cropped_image, dims=(-1,))
                    # run part regressor
                    f_part = self.Encoder[part](cropped_image)
                    part_dict = self.decompose_code(self.Regressor[part](
                        f_part), self.param_list_dict[f'{part}_list'])
                    part_share_dict = self.decompose_code(self.Regressor[f'{part}_share'](
                        f_part), self.param_list_dict[f'{part}_share_list'])
                    param_dict['body_' +
                               part_name] = {**part_dict, **part_share_dict}

                    # moderator to assign weight, then integrate features
                    f_body_out, f_part_out, f_weight = self.Moderator[f'{part}_share'](
                        feature['body'][f'{part_name}_share'], f_part, work=True)
                    if copy_and_paste:
                        # copy and paste strategy always trusts the results from part
                        feature['body'][f'{part_name}_share'] = f_part
                    elif threthold and part == 'hand':
                        # for hand, if part weight > 0.7 (very confident, then fully trust part)
                        part_w = f_weight[:, [1]]
                        part_w[part_w > 0.7] = 1.
                        f_body_out = feature['body'][f'{part_name}_share']*(
                            1. - part_w) + f_part*part_w
                        feature['body'][f'{part_name}_share'] = f_body_out
                    else:
                        feature['body'][f'{part_name}_share'] = f_body_out
                    fusion_weight[part_name] = f_weight
                # save weights from moderator, that can be further used for optimization/running specific tasks on parts
                param_dict['moderator_weight'] = fusion_weight

                # -- predict parameters from fused body feature
                # head share
                head_share_dict = self.decompose_code(self.Regressor['head'+'_share'](
                    feature[key]['head'+'_share']), self.param_list_dict['head'+'_share_list'])
                # right hand share
                right_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                    feature[key]['right_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                # left hand share
                left_hand_share_dict = self.decompose_code(self.Regressor['hand'+'_share'](
                    feature[key]['left_hand'+'_share']), self.param_list_dict['hand'+'_share_list'])
                # change the dict name from right to left
                left_hand_share_dict['left_hand_pose'] = left_hand_share_dict.pop(
                    'right_hand_pose')
                left_hand_share_dict['left_wrist_pose'] = left_hand_share_dict.pop(
                    'right_wrist_pose')
                param_dict['body'] = {
                    **body_dict, **head_share_dict, **left_hand_share_dict, **right_hand_share_dict}
                # copy tex param from head param dict to body param dict
                param_dict['body']['tex'] = param_dict['body_head']['tex']
                param_dict['body']['light'] = param_dict['body_head']['light']

                if keep_local:
                    # for local change that will not affect whole body and produce unnatral pose, trust part
                    param_dict[key]['exp'] = param_dict['body_head']['exp']
                    param_dict[key]['right_hand_pose'] = param_dict['body_right_hand']['right_hand_pose']
                    param_dict[key]['left_hand_pose'] = param_dict['body_left_hand']['right_hand_pose']

        return param_dict

    def convert_pose(self, param_dict, param_type):
        ''' Convert pose parameters to rotation matrix
        Args:
            param_dict: smplx parameters
            param_type: should be one of body/head/hand
        Returns:
            param_dict: smplx parameters 
        '''
        assert param_type in ['body', 'head', 'hand']

        # convert pose representations: the output from network are continous repre or axis angle,
        # while the input pose for smplx need to be rotation matrix
        for key in param_dict:
            if "pose" in key and 'jaw' not in key:
                param_dict[key] = converter.batch_cont2matrix(param_dict[key])
        if param_type == 'body' or param_type == 'head':
            param_dict['jaw_pose'] = converter.batch_euler2matrix(param_dict['jaw_pose'])[
                :, None, :, :]

        # complement params if it's not in given param dict
        if param_type == 'head':
            batch_size = param_dict['shape'].shape[0]
            param_dict['abs_head_pose'] = param_dict['head_pose'].clone()
            param_dict['global_pose'] = param_dict['head_pose']
            param_dict['partbody_pose'] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1)[:, :self.param_list_dict['body_list']['partbody_pose']]
            param_dict['neck_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_hand_pose'] = self.smplx.left_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['right_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['right_hand_pose'] = self.smplx.right_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
        elif param_type == 'hand':
            batch_size = param_dict['right_hand_pose'].shape[0]
            param_dict['abs_right_wrist_pose'] = param_dict['right_wrist_pose'].clone()
            dtype = param_dict['right_hand_pose'].dtype
            device = param_dict['right_hand_pose'].device
            x_180_pose = torch.eye(
                3, dtype=dtype, device=device).unsqueeze(0).repeat(1, 1, 1)
            x_180_pose[0, 2, 2] = -1.
            x_180_pose[0, 1, 1] = -1.
            param_dict['global_pose'] = x_180_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['shape'] = self.smplx.shape_params.expand(
                batch_size, -1)
            param_dict['exp'] = self.smplx.expression_params.expand(
                batch_size, -1)
            param_dict['head_pose'] = self.smplx.head_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['neck_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['jaw_pose'] = self.smplx.jaw_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['partbody_pose'] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1)[:, :self.param_list_dict['body_list']['partbody_pose']]
            param_dict['left_wrist_pose'] = self.smplx.neck_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
            param_dict['left_hand_pose'] = self.smplx.left_hand_pose.unsqueeze(
                0).expand(batch_size, -1, -1, -1)
        elif param_type == 'body':
            # the predcition from the head and hand share regressor is always absolute pose
            batch_size = param_dict['shape'].shape[0]
            param_dict['abs_head_pose'] = param_dict['head_pose'].clone()
            param_dict['abs_right_wrist_pose'] = param_dict['right_wrist_pose'].clone()
            param_dict['abs_left_wrist_pose'] = param_dict['left_wrist_pose'].clone()
            # the body-hand share regressor is working for right hand
            # so we assume body network get the flipped feature for the left hand. then get the parameters
            # then we need to flip it back to left, which matches the input left hand
            param_dict['left_wrist_pose'] = util.flip_pose(
                param_dict['left_wrist_pose'])
            param_dict['left_hand_pose'] = util.flip_pose(
                param_dict['left_hand_pose'])
        else:
            exit()

        return param_dict

    def decode(self, param_dict, param_type):
        ''' Decode model parameters to smplx vertices & joints & texture
        Args:
            param_dict: smplx parameters
            param_type: should be one of body/head/hand
        Returns:
            predictions: smplx predictions
        '''
        if 'jaw_pose' in param_dict.keys() and len(param_dict['jaw_pose'].shape) == 2:
            self.convert_pose(param_dict, param_type)
        elif param_dict['right_wrist_pose'].shape[-1] == 6:
            self.convert_pose(param_dict, param_type)

        # concatenate body pose
        partbody_pose = param_dict['partbody_pose']
        param_dict['body_pose'] = torch.cat(
            [partbody_pose[:, :11],
             param_dict['neck_pose'],
             partbody_pose[:, 11:11+2],
             param_dict['head_pose'],
             partbody_pose[:, 13:13+4],
             param_dict['left_wrist_pose'],
             param_dict['right_wrist_pose']], dim=1)

        # change absolute head&hand pose to relative pose according to rest body pose
        if param_type == 'head' or param_type == 'body':
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'], param_dict['body_pose'], abs_joint='head')
        if param_type == 'hand' or param_type == 'body':
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'], param_dict['body_pose'], abs_joint='left_wrist')
            param_dict['body_pose'] = self.smplx.pose_abs2rel(
                param_dict['global_pose'], param_dict['body_pose'], abs_joint='right_wrist')

        if self.cfg.model.check_pose:
            # check if pose is natural (relative rotation), if not, set relative to 0 (especially for head pose)
            # xyz: pitch(positive for looking down), yaw(positive for looking left), roll(rolling chin to left)
            for pose_ind in [14]:  # head [15-1, 20-1, 21-1]:
                curr_pose = param_dict['body_pose'][:, pose_ind]
                euler_pose = converter._compute_euler_from_matrix(curr_pose)
                for i, max_angle in enumerate([20, 70, 10]):
                    euler_pose_curr = euler_pose[:, i]
                    euler_pose_curr[euler_pose_curr != torch.clamp(
                        euler_pose_curr, min=-max_angle*np.pi/180, max=max_angle*np.pi/180)] = 0.
                param_dict['body_pose'][:,
                                        pose_ind] = converter.batch_euler2matrix(euler_pose)

        # SMPLX
        verts, landmarks, joints = self.smplx(
            shape_params=param_dict['shape'],
            expression_params=param_dict['exp'],
            global_pose=param_dict['global_pose'],
            body_pose=param_dict['body_pose'],
            jaw_pose=param_dict['jaw_pose'],
            left_hand_pose=param_dict['left_hand_pose'],
            right_hand_pose=param_dict['right_hand_pose'])
        smplx_kpt3d = joints.clone()

        # projection
        cam = param_dict[param_type + '_cam']
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks, cam)[:, :, :2]
        predicted_joints = util.batch_orth_proj(joints, cam)[:, :, :2]

        prediction = {
            'vertices': verts,
            'transformed_vertices': trans_verts,
            'face_kpt': predicted_landmarks,
            'smplx_kpt': predicted_joints,
            'smplx_kpt3d': smplx_kpt3d,
            'joints': joints,
            'cam': param_dict[param_type + '_cam']
        }

        # change the order of face keypoints, to be the same as "standard" 68 keypoints
        prediction['face_kpt'] = torch.cat(
            [prediction['face_kpt'][:, -17:], prediction['face_kpt'][:, :-17]], dim=1)

        prediction.update(param_dict)

        return prediction

    def decode_Tpose(self, param_dict):
        ''' return body mesh in T pose, support body and head param dict only
        '''
        verts, _, _ = self.smplx(
            shape_params=param_dict['shape'],
            expression_params=param_dict['exp'],
            jaw_pose=param_dict['jaw_pose'])
        return verts
