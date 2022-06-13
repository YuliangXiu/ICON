
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

from lib.hybrik.models.simple3dpose import HybrIKBaseSMPLCam
from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pixielib.pixie import PIXIE
import smplx
from lib.pare.pare.core.tester import PARETester
from lib.pymaf.utils.geometry import rotation_matrix_to_angle_axis, batch_rodrigues
from lib.pymaf.utils.imutils import process_image
from lib.pymaf.core import path_config
from lib.pymaf.models import pymaf_net
from lib.common.config import cfg
from lib.common.render import Render
from lib.dataset.body_model import TetraSMPLModel
from lib.dataset.mesh_util import get_visibility, SMPLX
import os.path as osp
import os
import torch
import glob
import numpy as np
import random
import human_det
from termcolor import colored
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset():
    def __init__(self, cfg, device):

        random.seed(1993)

        self.image_dir = cfg['image_dir']
        self.seg_dir = cfg['seg_dir']
        self.has_det = cfg['has_det']
        self.hps_type = cfg['hps_type']
        self.smpl_type = 'smpl' if cfg['hps_type'] != 'pixie' else 'smplx'
        self.smpl_gender = 'neutral'

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

        # smpl related
        self.smpl_data = SMPLX()

        self.get_smpl_model = lambda smpl_type, smpl_gender: smplx.create(
            model_path=self.smpl_data.model_dir,
            gender=smpl_gender,
            model_type=smpl_type,
            ext='npz')

        # Load SMPL model
        self.smpl_model = self.get_smpl_model(
            self.smpl_type, self.smpl_gender).to(self.device)
        self.faces = self.smpl_model.faces

        if self.hps_type == 'pymaf':
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS,
                                 pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(
                path_config.CHECKPOINT_FILE)['model'],
                strict=True)
            self.hps.eval()

        elif self.hps_type == 'pare':
            self.hps = PARETester(path_config.CFG, path_config.CKPT).model
        elif self.hps_type == 'pixie':
            self.hps = PIXIE(config=pixie_cfg, device=self.device)
            self.smpl_model = self.hps.smplx
        elif self.hps_type == 'hybrik':
            smpl_path = osp.join(self.smpl_data.model_dir,
                                 "smpl/SMPL_NEUTRAL.pkl")
            self.hps = HybrIKBaseSMPLCam(
                cfg_file=path_config.HYBRIK_CFG, smpl_path=smpl_path, data_path=path_config.hybrik_data_dir)
            self.hps.load_state_dict(torch.load(
                path_config.HYBRIK_CKPT, map_location='cpu'), strict=False)
            self.hps.to(self.device)
        elif self.hps_type == 'bev':
            try:
                import bev
            except:
                print(
                    'Could not find bev, installing via pip install --upgrade simple-romp')
                os.system('pip install simple-romp==1.0.3')
                import bev
            settings = bev.main.default_settings
            # change the argparse settings of bev here if you prefer other settings.
            settings.mode = 'image'
            settings.GPU = int(str(self.device).split(':')[1])
            settings.show_largest = True
            # settings.show = True # uncommit this to show the original BEV predictions
            self.hps = bev.BEV(settings)

        print(colored(f"Using {self.hps_type} as HPS Estimator\n", "green"))

        self.render = Render(size=512, device=device)

    def __len__(self):
        return len(self.subject_list)

    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        smpl_verts *= torch.tensor([1.0, -1.0, 1.0]).to(self.device)
        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        if self.smpl_type == 'smpl':
            smplx_ind = self.smpl_data.smpl2smplx(np.arange(smpl_vis.shape[0]))
        else:
            smplx_ind = np.arange(smpl_vis.shape[0])
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

        if self.seg_dir is None:
            img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(
                img_path, self.det, self.hps_type, 512, self.device)

            data_dict = {
                'name': img_name,
                'image': img_icon.to(self.device).unsqueeze(0),
                'ori_image': img_ori,
                'mask': img_mask,
                'uncrop_param': uncrop_param
            }

        else:
            img_icon, img_hps, img_ori, img_mask, uncrop_param, segmentations = process_image(
                img_path, self.det, self.hps_type, 512, self.device,
                seg_path=os.path.join(self.seg_dir, f'{img_name}.json'))
            data_dict = {
                'name': img_name,
                'image': img_icon.to(self.device).unsqueeze(0),
                'ori_image': img_ori,
                'mask': img_mask,
                'uncrop_param': uncrop_param,
                'segmentations': segmentations
            }

        with torch.no_grad():
            # import ipdb; ipdb.set_trace()
            preds_dict = self.hps.forward(img_hps)

        data_dict['smpl_faces'] = torch.Tensor(
            self.faces.astype(np.int16)).long().unsqueeze(0).to(
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

        elif self.hps_type == 'pixie':
            data_dict.update(preds_dict)
            data_dict['body_pose'] = preds_dict['body_pose']
            data_dict['global_orient'] = preds_dict['global_pose']
            data_dict['betas'] = preds_dict['shape']
            data_dict['smpl_verts'] = preds_dict['vertices']
            scale, tranX, tranY = preds_dict['cam'][0, :3]

        elif self.hps_type == 'hybrik':
            data_dict['body_pose'] = preds_dict['pred_theta_mats'][:, 1:]
            data_dict['global_orient'] = preds_dict['pred_theta_mats'][:, [0]]
            data_dict['betas'] = preds_dict['pred_shape']
            data_dict['smpl_verts'] = preds_dict['pred_vertices']
            scale, tranX, tranY = preds_dict['pred_camera'][0, :3]
            scale = scale * 2

        elif self.hps_type == 'bev':
            data_dict['betas'] = torch.from_numpy(preds_dict['smpl_betas'])[
                [0], :10].to(self.device).float()
            pred_thetas = batch_rodrigues(torch.from_numpy(
                preds_dict['smpl_thetas'][0]).reshape(-1, 3)).float()
            data_dict['body_pose'] = pred_thetas[1:][None].to(self.device)
            data_dict['global_orient'] = pred_thetas[[0]][None].to(self.device)
            data_dict['smpl_verts'] = torch.from_numpy(
                preds_dict['verts'][[0]]).to(self.device).float()
            tranX = preds_dict['cam_trans'][0, 0]
            tranY = preds_dict['cam'][0, 1] + 0.28
            scale = preds_dict['cam'][0, 0] * 1.1

        data_dict['scale'] = scale
        data_dict['trans'] = torch.tensor(
            [tranX, tranY, 0.0]).to(self.device).float()

        # data_dict info (key-shape):
        # scale, tranX, tranY - tensor.float
        # betas - [1,10] / [1, 200]
        # body_pose - [1, 23, 3, 3] / [1, 21, 3, 3]
        # global_orient - [1, 1, 3, 3]
        # smpl_verts - [1, 6890, 3] / [1, 10475, 3]

        return data_dict

    def render_normal(self, verts, faces, deform_verts=None):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_simple_mesh(verts, faces, deform_verts)
        return self.render.get_clean_image()

    def visualize_alignment(self, data):

        import vedo
        import trimesh

        if self.hps_type != 'pixie':
            smpl_out = self.smpl_model(betas=data['betas'],
                                       body_pose=data['body_pose'],
                                       global_orient=data['global_orient'],
                                       pose2rot=False)
            smpl_verts = (
                (smpl_out.vertices + data['trans']) * data['scale']).detach().cpu().numpy()[0]
        else:
            smpl_verts, _, _ = self.smpl_model(shape_params=data['betas'],
                                               expression_params=data['exp'],
                                               body_pose=data['body_pose'],
                                               global_pose=data['global_orient'],
                                               jaw_pose=data['jaw_pose'],
                                               left_hand_pose=data['left_hand_pose'],
                                               right_hand_pose=data['right_hand_pose'])

            smpl_verts = (
                (smpl_verts + data['trans']) * data['scale']).detach().cpu().numpy()[0]

        smpl_verts *= np.array([1.0, -1.0, -1.0])
        faces = data['smpl_faces'][0].detach().cpu().numpy()

        image_P = data['image']
        image_F, image_B = self.render_normal(smpl_verts, faces)

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500))
        vis_list = []

        image_F = (
            0.5 * (1.0 + image_F[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
        image_B = (
            0.5 * (1.0 + image_B[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
        image_P = (
            0.5 * (1.0 + image_P[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)

        vis_list.append(vedo.Picture(image_P*0.5+image_F *
                        0.5).scale(2.0/image_P.shape[0]).pos(-1.0, -1.0, 1.0))
        vis_list.append(vedo.Picture(image_F).scale(
            2.0/image_F.shape[0]).pos(-1.0, -1.0, -0.5))
        vis_list.append(vedo.Picture(image_B).scale(
            2.0/image_B.shape[0]).pos(-1.0, -1.0, -1.0))

        # create a mesh
        mesh = trimesh.Trimesh(smpl_verts, faces, process=False)
        mesh.visual.vertex_colors = [200, 200, 0]
        vis_list.append(mesh)

        vp.show(*vis_list, bg="white", axes=1, interactive=True)


if __name__ == '__main__':

    cfg.merge_from_file("./configs/icon-filter.yaml")
    cfg.merge_from_file('./lib/pymaf/configs/pymaf_config.yaml')

    cfg_show_list = [
        'test_gpus', ['0'], 'mcube_res', 512, 'clean_mesh', False
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda:0')

    dataset = TestDataset(
        {
            'image_dir': "./examples",
            'has_det': True,    # w/ or w/o detection
            'hps_type': 'bev'  # pymaf/pare/pixie/hybrik/bev
        }, device)

    for i in range(len(dataset)):
        dataset.visualize_alignment(dataset[i])
