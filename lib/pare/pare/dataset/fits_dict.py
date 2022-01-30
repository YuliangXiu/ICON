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
import torch
import numpy as np
from loguru import logger
from torchgeometry import angle_axis_to_rotation_matrix

from ..core import constants, config


class FitsDict():
    """ Dictionary keeping track of the best fit per image in the training set """
    def __init__(self, hparams, train_dataset):
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.fits_dict = {}
        # array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(constants.SMPL_POSE_FLIP_PERM,
                                          dtype=torch.int64)
        # Load dictionary state
        for ds_name, ds in train_dataset.dataset_dict.items():
            try:
                dict_file = os.path.join(self.hparams.LOG_DIR,
                                         ds_name + '_fits.npy')
                self.fits_dict[ds_name] = torch.from_numpy(np.load(dict_file))
            except IOError:
                # Dictionary does not exist, so populate with static fits
                dict_file = os.path.join(config.STATIC_FITS_DIR,
                                         ds_name + '_fits.npy')
                try:
                    self.fits_dict[ds_name] = torch.from_numpy(
                        np.load(dict_file))
                except IOError:
                    logger.warning(
                        f'Static fits for {ds_name} could not found, initializing from mosh params'
                    )
                    h36m = np.load('data/dataset_extras/h36m_train.npz')
                    self.fits_dict[ds_name] = torch.from_numpy(
                        np.hstack([h36m['pose'], h36m['shape']]))

    def save(self):
        """ Save dictionary state to disk """
        for ds_name in self.train_dataset.dataset_dict.keys():
            dict_file = os.path.join(self.hparams.LOG_DIR,
                                     ds_name + '_fits.npy')
            np.save(dict_file, self.fits_dict[ds_name].cpu().numpy())

    def __getitem__(self, x):
        """ Retrieve dictionary entries """
        dataset_name, ind, rot, is_flipped = x
        batch_size = len(dataset_name)
        pose = torch.zeros((batch_size, 72))
        betas = torch.zeros((batch_size, 10))
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            params = self.fits_dict[ds][i]
            pose[n, :] = params[:72]
            betas[n, :] = params[72:]
        pose = pose.clone()
        # Apply flipping and rotation
        pose = self.flip_pose(self.rotate_pose(pose, rot), is_flipped)
        betas = betas.clone()
        return pose, betas

    def __setitem__(self, x, val):
        """ Update dictionary entries """
        dataset_name, ind, rot, is_flipped, update = x
        pose, betas = val
        batch_size = len(dataset_name)
        # Undo flipping and rotation
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped), -rot)
        params = torch.cat((pose, betas), dim=-1).cpu()
        for ds, i, n in zip(dataset_name, ind, range(batch_size)):
            if update[n]:
                self.fits_dict[ds][i] = params[n]

    def flip_pose(self, pose, is_flipped):
        """flip SMPL pose parameters"""
        is_flipped = is_flipped.bool()
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, self.flipped_parts]
        # we also negate the second and the third dimension of the axis-angle representation
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees"""
        pose = pose.clone()
        cos = torch.cos(-np.pi * rot / 180.)
        sin = torch.sin(-np.pi * rot / 180.)
        zeros = torch.zeros_like(cos)
        r3 = torch.zeros(cos.shape[0], 1, 3, device=cos.device)
        r3[:, 0, -1] = 1
        R = torch.cat([
            torch.stack([cos, -sin, zeros], dim=-1).unsqueeze(1),
            torch.stack([sin, cos, zeros], dim=-1).unsqueeze(1), r3
        ],
                      dim=1)
        global_pose = pose[:, :3]
        global_pose_rotmat = angle_axis_to_rotation_matrix(global_pose)
        global_pose_rotmat_3b3 = global_pose_rotmat[:, :3, :3]
        global_pose_rotmat_3b3 = torch.matmul(R, global_pose_rotmat_3b3)
        global_pose_rotmat[:, :3, :3] = global_pose_rotmat_3b3
        global_pose_rotmat = global_pose_rotmat[:, :-1, :-1].cpu().numpy()
        global_pose_np = np.zeros((global_pose.shape[0], 3))
        for i in range(global_pose.shape[0]):
            aa, _ = cv2.Rodrigues(global_pose_rotmat[i])
            global_pose_np[i, :] = aa.squeeze()
        pose[:, :3] = torch.from_numpy(global_pose_np).to(pose.device)
        return pose
