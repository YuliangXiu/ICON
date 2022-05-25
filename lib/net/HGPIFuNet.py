
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

from lib.net.voxelize import Voxelization
from lib.dataset.mesh_util import cal_sdf_batch, feat_select, read_smpl_constants
from lib.net.NormalNet import NormalNet
from lib.net.MLP import MLP
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.HGFilters import *
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
import os


maskout = False


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 cfg,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss()):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode,
                                        error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim
        self.sdf_clip = cfg.sdf_clip / 100.0

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        if self.prior_type == 'icon':
            if 'image' in self.in_geo:
                self.channels_filter = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8]]
            else:
                self.channels_filter = [[0, 1, 2], [3, 4, 5]]

        else:
            if 'image' in self.in_geo:
                self.channels_filter = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
            else:
                self.channels_filter = [[0, 1, 2, 3, 4, 5]]

        channels_IF[0] = self.hourglass_dim if self.use_filter else len(
            self.channels_filter[0])

        if self.prior_type == 'icon' and 'vis' not in self.smpl_feats:
            if self.use_filter:
                channels_IF[0] += self.hourglass_dim
            else:
                channels_IF[0] += len(self.channels_filter[0])

        if self.prior_type == 'icon':
            channels_IF[0] += self.smpl_dim
        elif self.prior_type == 'pamir':
            channels_IF[0] += self.voxel_dim
            smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = read_smpl_constants(
                self.smplx_data.tedra_dir)
            self.voxelization = Voxelization(
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=cfg.batch_size,
                device=torch.device(f"cuda:{cfg.gpus[0]}"))
            self.ve = VolumeEncoder(3, self.voxel_dim, self.opt.num_stack)
        else:
            channels_IF[0] += 1

        self.icon_keys = ["smpl_verts", "smpl_faces", "smpl_vis", "smpl_cmap"]
        self.pamir_keys = [
            "voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"
        ]

        self.if_regressor = MLP(
            filter_channels=channels_IF,
            name='if',
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None)

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack,
                                         len(self.channels_filter[0]))
            else:
                print(
                    colored(f"Backbone {self.opt.gtype} is unimplemented",
                            'green'))

        summary_log = f"{self.prior_type.upper()}:\n" + \
            f"w/ Global Image Encoder: {self.use_filter}\n" + \
            f"Image Features used by MLP: {self.in_geo}\n"

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): 6\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(cfg)
        init_net(self)

    def get_normal(self, in_tensor_dict):

        # insert normal features
        if (not self.training) and (not self.overfit):
            # print(colored("infer normal","blue"))
            with torch.no_grad():
                feat_lst = []
                if "image" in self.in_geo:
                    feat_lst.append(
                        in_tensor_dict['image'])  # [1, 3, 512, 512]
                if 'normal_F' in self.in_geo and 'normal_B' in self.in_geo:
                    if 'normal_F' not in in_tensor_dict.keys(
                    ) or 'normal_B' not in in_tensor_dict.keys():
                        (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                    else:
                        nmlF = in_tensor_dict['normal_F']
                        nmlB = in_tensor_dict['normal_B']
                    feat_lst.append(nmlF)  # [1, 3, 512, 512]
                    feat_lst.append(nmlB)  # [1, 3, 512, 512]
            in_filter = torch.cat(feat_lst, dim=1)

        else:
            in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo],
                                  dim=1)

        return in_filter

    def get_mask(self, in_filter, size=128):

        mask = F.interpolate(in_filter[:, self.channels_filter[0]],
                             size=(size, size),
                             mode="bilinear",
                             align_corners=True).abs().sum(dim=1,
                                                           keepdim=True) != 0.0

        return mask

    def filter(self, in_tensor_dict, return_inter=False):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''

        in_filter = self.get_normal(in_tensor_dict)

        features_G = []

        if self.prior_type == 'icon':
            if self.use_filter:
                features_F = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]]
                                           )  # [(B,hg_dim,128,128) * 4]
                features_B = self.F_filter(in_filter[:,
                                                     self.channels_filter[1]]
                                           )  # [(B,hg_dim,128,128) * 4]
            else:
                features_F = [in_filter[:, self.channels_filter[0]]]
                features_B = [in_filter[:, self.channels_filter[1]]]
            for idx in range(len(features_F)):
                features_G.append(
                    torch.cat([features_F[idx], features_B[idx]], dim=1))
        else:
            if self.use_filter:
                features_G = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]])
            else:
                features_G = [in_filter[:, self.channels_filter[0]]]

        if self.prior_type == 'icon':
            self.smpl_feat_dict = {
                k: in_tensor_dict[k]
                for k in self.icon_keys
            }
        elif self.prior_type == "pamir":
            self.smpl_feat_dict = {
                k: in_tensor_dict[k]
                for k in self.pamir_keys
            }
        else:
            pass
            # print(colored("use z rather than icon or pamir", "green"))

        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = [features_G[-1]]
        else:
            features_out = features_G

        if maskout:
            features_out_mask = []
            for feat in features_out:
                features_out_mask.append(
                    feat * self.get_mask(in_filter, size=feat.shape[2]))
            features_out = features_out_mask

        if return_inter:
            return features_out, in_filter
        else:
            return features_out

    def query(self, features, points, calibs, transforms=None, regressor=None):

        xyz = self.projection(points, calibs, transforms)

        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []

        if self.prior_type == 'icon':

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # points [B, 3, N]

            smpl_sdf, smpl_norm, smpl_cmap, smpl_vis = cal_sdf_batch(
                self.smpl_feat_dict['smpl_verts'],
                self.smpl_feat_dict['smpl_faces'],
                self.smpl_feat_dict['smpl_cmap'],
                self.smpl_feat_dict['smpl_vis'],
                xyz.permute(0, 2, 1).contiguous())

            # smpl_sdf [B, N, 1]
            # smpl_norm [B, N, 3]
            # smpl_cmap [B, N, 3]
            # smpl_vis [B, N, 1]

            # set ourlier point features as uniform values
            smpl_outlier = torch.abs(smpl_sdf).ge(self.sdf_clip)
            smpl_sdf[smpl_outlier] = torch.sign(smpl_sdf[smpl_outlier])

            feat_lst = [smpl_sdf]
            if 'cmap' in self.smpl_feats:
                smpl_cmap[smpl_outlier.repeat(
                    1, 1, 3)] = smpl_sdf[smpl_outlier].repeat(1, 1, 3)
                feat_lst.append(smpl_cmap)
            if 'norm' in self.smpl_feats:
                feat_lst.append(smpl_norm)
            if 'vis' in self.smpl_feats:
                feat_lst.append(smpl_vis)

            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)
            vol_feats = features

        elif self.prior_type == "pamir":

            voxel_verts = self.smpl_feat_dict[
                'voxel_verts'][:, :-self.smpl_feat_dict['pad_v_num'][0], :]
            voxel_faces = self.smpl_feat_dict[
                'voxel_faces'][:, :-self.smpl_feat_dict['pad_f_num'][0], :]

            self.voxelization.update_param(
                batch_size=voxel_faces.shape[0],
                smpl_tetra=voxel_faces[0].detach().cpu().numpy())
            vol = self.voxelization(voxel_verts)  # vol ~ [0,1]
            vol_feats = self.ve(vol, intermediate_output=self.training)
        else:
            vol_feats = features

        for im_feat, vol_feat in zip(features, vol_feats):

            # [B, Feat_i + z, N]
            # normal feature choice by smpl_vis
            if self.prior_type == 'icon':
                if 'vis' in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy),
                                                   smpl_feat[:, [-1], :])
                    if maskout:
                        normal_mask = torch.tile(
                            point_local_feat.sum(dim=1, keepdims=True) == 0.0,
                            (1, smpl_feat.shape[1], 1))
                        normal_mask[:, 1:, :] = False
                        smpl_feat[normal_mask] = -1.0
                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :]]

            elif self.prior_type == 'pamir':
                # im_feat [B, hg_dim, 128, 128]
                # vol_feat [B, vol_dim, 32, 32, 32]
                point_feat_list = [
                    self.index(im_feat, xy),
                    self.index(vol_feat, xyz)
                ]

            else:
                point_feat_list = [self.index(im_feat, xy), z]

            point_feat = torch.cat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat)
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def get_error(self, preds_if_list, labels):
        """calcaulate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += self.error_term(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if

    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """

        sample_tensor = in_tensor_dict['sample']
        calib_tensor = in_tensor_dict['calib']
        label_tensor = in_tensor_dict['label']

        in_feat = self.filter(in_tensor_dict)

        preds_if_list = self.query(in_feat,
                                   sample_tensor,
                                   calib_tensor,
                                   regressor=self.if_regressor)

        error = self.get_error(preds_if_list, label_tensor)

        return preds_if_list[-1], error
