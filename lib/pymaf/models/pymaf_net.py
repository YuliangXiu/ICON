import torch
import torch.nn as nn
import numpy as np

from lib.pymaf.utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .res_module import IUV_predict_layer
from lib.common.config import cfg
import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=1,
                J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(betas=pred_shape,
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3,
                                                                 3)).reshape(
                                                                     -1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta': torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_kp_3d': pred_smpl_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self,
                     x,
                     init_pose=None,
                     init_shape=None,
                     init_cam=None,
                     n_iter=1,
                     J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(
            batch_size, 24, 3, 3)

        pred_output = self.smpl(betas=pred_shape,
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3,
                                                                 3)).reshape(
                                                                     -1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta': torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_kp_3d': pred_smpl_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output


class PyMAF(nn.Module):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True):
        super().__init__()
        self.feature_extractor = ResNet_Backbone(
            model=cfg.MODEL.PyMAF.BACKBONE, pretrained=pretrained)

        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            self.maf_extractor.append(MAF_Extractor())
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[
            0] * cfg.MODEL.PyMAF.MLP_DIM[-1]

        grid_size = 21
        xv, yv = torch.meshgrid([
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size)
        ])
        points_grid = torch.stack([xv.reshape(-1),
                                   yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            self.regressor.append(
                Regressor(feat_dim=ref_infeat_dim,
                          smpl_mean_params=smpl_mean_params))

        dp_feat_dim = 256
        self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(
                num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(in_channels=self.inplanes,
                                   out_channels=planes,
                                   kernel_size=kernel,
                                   stride=2,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, J_regressor=None):

        batch_size = x.shape[0]

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(x)

        assert cfg.MODEL.PyMAF.N_ITER >= 0 and cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.PyMAF.N_ITER == 3:
            deconv_blocks = [
                self.deconv_layers[0:3], self.deconv_layers[3:6],
                self.deconv_layers[6:9]
            ]

        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat,
                                                     J_regressor=J_regressor)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())

            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(
                    self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)
            else:
                pred_smpl_verts = smpl_output['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(
                    self.maf_extractor[rf_i].Dmap.unsqueeze(0),
                    pred_smpl_verts)  # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i](
                    pred_smpl_verts_ds)  # [B, 431 * n_feat]

            smpl_output = self.regressor[rf_i](ref_feature,
                                               pred_pose,
                                               pred_shape,
                                               pred_cam,
                                               n_iter=1,
                                               J_regressor=J_regressor)
            out_list['smpl_out'].append(smpl_output)

        if self.training and cfg.MODEL.PyMAF.AUX_SUPV_ON:
            iuv_out_dict = self.dp_head(s_feat)
            out_list['dp_out'].append(iuv_out_dict)

        return out_list


def pymaf_net(smpl_mean_params, pretrained=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params, pretrained)
    return model
