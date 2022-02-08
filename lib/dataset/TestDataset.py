
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

import os.path as osp
import os
import torch
import glob
import numpy as np
import sys
import random
import human_det
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# project related libs
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.dataset.mesh_util import get_visibility, SMPLX
from lib.dataset.body_model import TetraSMPLModel
from lib.common.render import Render
from lib.common.config import cfg

# for pymaf
from lib.pymaf.models import SMPL, pymaf_net
from lib.pymaf.core import path_config
from lib.pymaf.utils.imutils import process_image
from lib.pymaf.utils.geometry import rotation_matrix_to_angle_axis

# for pare
from lib.pare.pare.core.tester import PARETester


class TestDataset():
    def __init__(self, cfg, device):

        random.seed(1993)

        self.image_dir = cfg['image_dir']
        self.has_det = cfg['has_det']
        self.hps_type = cfg['hps_type']
        self.device = device

        if self.has_det:
            self.det = human_det.Detection()
        else:
            self.det = None

        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ['jpg', 'png', 'jpeg', "JPG", 'bmp']
        keep_lst = [
            item for item in keep_lst if item.split(".")[-1] in img_fmts
        ]

        self.subject_list = sorted(
            [item for item in keep_lst if item.split(".")[-1] in img_fmts])

        self.smpl_data = SMPLX()
        if self.hps_type == 'pymaf':
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS,
                                 pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(
                path_config.CHECKPOINT_FILE)['model'],
                                     strict=True)
            self.hps.eval()
            
        # uncomment if you want to switch HPS to PARE
        elif self.hps_type == 'pare':
            self.hps = PARETester(path_config.CFG, path_config.CKPT).model

        # Load SMPL model
        self.smpl_model = SMPL(osp.join(self.smpl_data.model_dir, "smpl"),
                               batch_size=1,
                               create_transl=False).to(self.device)

        self.render = Render(size=512, device=device)

    def __len__(self):
        return len(self.subject_list)

    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        smpl_verts *= torch.tensor([1.0, -1.0, 1.0]).to(self.device)
        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        smplx_ind = self.smpl_data.smpl2smplx(np.arange(smpl_vis.shape[0]))
        smpl_cmap = self.smpl_data.get_smpl_mat(smplx_ind)

        return {
            'smpl_vis': smpl_vis.unsqueeze(0).to(self.device),
            'smpl_cmap': smpl_cmap.unsqueeze(0).to(self.device),
            'smpl_verts': smpl_verts.unsqueeze(0)
        }

    def compute_voxel_verts(self, body_pose, global_orient, betas, trans,
                            scale):

        smpl_path = osp.join(self.smpl_data.model_dir, "smpl/SMPL_NEUTRAL.pkl")
        tetra_path = osp.join(self.smpl_data.tedra_dir,
                              'tetra_neutral_adult_smpl.npz')
        smpl_model = TetraSMPLModel(smpl_path, tetra_path, 'adult')

        pose = torch.cat([global_orient[0], body_pose[0]], dim=0)
        smpl_model.set_params(rotation_matrix_to_angle_axis(pose),
                              beta=betas[0])

        verts = np.concatenate(
            [smpl_model.verts, smpl_model.verts_added],
            axis=0) * scale.item() + trans.detach().cpu().numpy()
        faces = np.loadtxt(osp.join(self.smpl_data.tedra_dir,
                                    'tetrahedrons_neutral_adult.txt'),
                           dtype=np.int32) - 1

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = np.pad(verts, ((0, pad_v_num), (0, 0)),
                       mode='constant',
                       constant_values=0.0).astype(np.float32) * 0.5
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)),
                       mode='constant',
                       constant_values=0.0).astype(np.int32)

        verts[:, 2] *= -1.0

        voxel_dict = {
            'voxel_verts':
            torch.from_numpy(verts).to(self.device).unsqueeze(0).float(),
            'voxel_faces':
            torch.from_numpy(faces).to(self.device).unsqueeze(0).long(),
            'pad_v_num':
            torch.tensor(pad_v_num).to(self.device).unsqueeze(0).long(),
            'pad_f_num':
            torch.tensor(pad_f_num).to(self.device).unsqueeze(0).long()
        }

        return voxel_dict

    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]
        img_icon, img_norm, img_np = process_image(img_path, self.det, 512)
        data_dict = {
            'name': img_name,
            'image': img_icon.to(self.device).unsqueeze(0),
            'ori_image': img_np
        }
        with torch.no_grad():
            preds_dict = self.hps(img_norm.to(self.device))

        data_dict['smpl_faces'] = torch.Tensor(
            self.smpl_model.faces.astype(np.int16)).long().unsqueeze(0).to(
                self.device)

        if self.hps_type == 'pymaf':
            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            data_dict['betas'] = output['pred_shape']
            data_dict['body_pose'] = output['rotmat'][:, 1:]
            data_dict['global_orient'] = output['rotmat'][:, 0:1]
            data_dict['smpl_verts'] = output['verts']

        elif self.hps_type == 'pare':
            data_dict['body_pose'] = preds_dict['pred_pose'][:, 1:]
            data_dict['global_orient'] = preds_dict['pred_pose'][:, 0:1]
            data_dict['betas'] = preds_dict['pred_shape']
            data_dict['smpl_verts'] = preds_dict['smpl_vertices']
            scale, tranX, tranY = preds_dict['pred_cam'][0, :3]

        trans = torch.tensor([tranX, tranY, 0.0]).to(self.device)
        data_dict['scale'] = scale
        data_dict['trans'] = trans

        return data_dict

    def render_normal(self, verts, faces, deform_verts=None):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_simple_mesh(verts, faces, deform_verts)
        return self.render.get_clean_image()
