# This script is borrowed from https://github.com/nkolot/SPIN/blob/master/models/smpl.py

import torch
import numpy as np
from lib.smplx import SMPL as _SMPL
from lib.smplx.body_models import ModelOutput
from lib.smplx.lbs import vertices2joints
from collections import namedtuple

from lib.pymaf.core import path_config, constants

SMPL_MEAN_PARAMS = path_config.SMPL_MEAN_PARAMS
SMPL_MODEL_DIR = path_config.SMPL_MODEL_DIR

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(path_config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.ModelOutput = namedtuple(
            'ModelOutput_', ModelOutput._fields + (
                'smpl_joints',
                'joints_J19',
            ))
        self.ModelOutput.__new__.__defaults__ = (None, ) * len(
            self.ModelOutput._fields)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super().forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        vertices = smpl_output.vertices
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        smpl_joints = smpl_output.joints[:, :24]
        joints = joints[:, self.joint_map, :]  # [B, 49, 3]
        joints_J24 = joints[:, -24:, :]
        joints_J19 = joints_J24[:, constants.J24_TO_J19, :]
        output = self.ModelOutput(vertices=vertices,
                                  global_orient=smpl_output.global_orient,
                                  body_pose=smpl_output.body_pose,
                                  joints=joints,
                                  joints_J19=joints_J19,
                                  smpl_joints=smpl_joints,
                                  betas=smpl_output.betas,
                                  full_pose=smpl_output.full_pose)
        return output


def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces


def get_part_joints(smpl_joints):
    batch_size = smpl_joints.shape[0]

    # part_joints = torch.zeros().to(smpl_joints.device)

    one_seg_pairs = [(0, 1), (0, 2), (0, 3), (3, 6), (9, 12), (9, 13), (9, 14),
                     (12, 15), (13, 16), (14, 17)]
    two_seg_pairs = [(1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19),
                     (18, 20), (19, 21)]

    one_seg_pairs.extend(two_seg_pairs)

    single_joints = [(10), (11), (15), (22), (23)]

    part_joints = []

    for j_p in one_seg_pairs:
        new_joint = torch.mean(smpl_joints[:, j_p], dim=1, keepdim=True)
        part_joints.append(new_joint)

    for j_p in single_joints:
        part_joints.append(smpl_joints[:, j_p:j_p + 1])

    part_joints = torch.cat(part_joints, dim=1)

    return part_joints
