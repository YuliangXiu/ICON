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

import time
import torch
import sys, os
import numpy as np
from os.path import join
from loguru import logger
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
# from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

from ..models import SMPL
from ..core import constants, config
from ..core.config import DATASET_FILES, DATASET_FOLDERS, EVAL_MESH_DATASETS
from ..utils.image_utils import crop, flip_img, flip_pose, flip_kp, transform, \
    rot_aa, random_crop, read_img
from .coco_occlusion import load_coco_occluders, load_pascal_occluders, occlude_with_pascal_objects, occlude_with_coco_objects
# from lib.pare.utils import kp_utils
from ..utils.geometry import batch_rot2aa, batch_rodrigues

# from ..utils.image_utils import crop_cv2 as crop


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """
    def __init__(self,
                 options,
                 dataset,
                 ignore_3d=False,
                 use_augmentation=True,
                 is_train=True,
                 num_images=0,
                 occluders=None):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                       std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        self.dataset_dict = {dataset: 0}

        if num_images > 0:
            if is_train:
                # select a random subset of the dataset
                rand = np.random.randint(0,
                                         len(self.imgname),
                                         size=(num_images))
                logger.info(
                    f'{rand.shape[0]} images are randomly sampled from {self.dataset}'
                )
                self.imgname = self.imgname[rand]
                self.data_subset = {}
                for f in self.data.files:
                    self.data_subset[f] = self.data[f][rand]
                self.data = self.data_subset
            else:
                interval = len(self.imgname) // num_images
                logger.info(
                    f'{len(self.imgname[::interval])} images are selected from {self.dataset}'
                )
                self.imgname = self.imgname[::interval]
                self.data_subset = {}
                for f in self.data.files:
                    self.data_subset[f] = self.data[f][::interval]
                self.data = self.data_subset

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            if 'pose_0yaw_inverseyz' in self.data:
                self.pose = self.data['pose_0yaw_inverseyz'].astype(np.float)
            else:
                self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        if dataset in [
                'agora-cam-v2', 'coco-cam-camreg', 'coco-campose-camreg'
        ] and is_train:
            self.cam_rotmat = self.data['cam_rotmat']

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt],
                                        axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1
                                    for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        self.occluders = None
        if is_train and self.options.USE_SYNTHETIC_OCCLUSION:
            if occluders is None:
                logger.info(f'Loading synthetic occluders for {dataset}...')
                if self.options.OCC_AUG_DATASET == 'coco':
                    self.occluders = load_coco_occluders()
                    logger.info(
                        f'Found {len(self.occluders["obj_class"])} suitable '
                        f'objects from {self.options.OCC_AUG_DATASET} dataset')
                elif self.options.OCC_AUG_DATASET == 'pascal':
                    self.occluders = load_pascal_occluders(
                        pascal_voc_root_path=config.PASCAL_ROOT)
                    logger.info(
                        f'Found {len(self.occluders)} suitable '
                        f'objects from {self.options.OCC_AUG_DATASET} dataset')
            else:
                logger.info('Using mixed/eft dataset occluders')
                self.occluders = occluders

        # evaluation variables
        if not self.is_train or (self.is_train and dataset
                                 == '3dpw') or self.options.NONPARAMETRIC:
            self.joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(
                np.load(config.JOINT_REGRESSOR_H36M)).float()

            self.smpl = SMPL(config.SMPL_MODEL_DIR,
                             batch_size=1,
                             create_transl=False)

            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False)

        self.length = self.scale.shape[0]
        logger.info(
            f'Loaded {self.dataset} dataset, num samples {self.length}')

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= self.options.FLIP_PROB:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.NOISE_FACTOR,
                                   1 + self.options.NOISE_FACTOR, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * self.options.ROT_FACTOR,
                max(-2 * self.options.ROT_FACTOR,
                    np.random.randn() * self.options.ROT_FACTOR))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + self.options.SCALE_FACTOR,
                max(1 - self.options.SCALE_FACTOR,
                    np.random.randn() * self.options.SCALE_FACTOR + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self,
                       rgb_img,
                       center,
                       scale,
                       rot,
                       flip,
                       pn,
                       img_res,
                       kp2d=None):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, [img_res, img_res], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)

        if self.occluders is not None:
            if self.options.OCC_AUG_DATASET == 'coco':
                rgb_img = occlude_with_coco_objects(
                    rgb_img,
                    kp2d=kp2d.copy(),
                    occluders=self.occluders,
                    img_size=self.options.IMG_RES)
            elif self.options.OCC_AUG_DATASET == 'pascal':
                rgb_img = occlude_with_pascal_objects(rgb_img, self.occluders)

        if self.is_train:  # and not self.dataset in ['ochuman', 'crowdpose']:
            # if not self.dataset in ['ochuman', 'crowdpose']:
            #     if time.localtime().tm_mday > 28 and time.localtime().tm_mday >=10:
            #         raise ValueError('remove ochuman, crowdpose exception from the base dataset!!!')
            albumentation_aug = A.Compose(transforms=[A.MotionBlur(p=0.5)])
            rgb_img = albumentation_aug(image=rgb_img)['image']

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :,
                0] = np.minimum(255.0, np.maximum(0.0,
                                                  rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :,
                1] = np.minimum(255.0, np.maximum(0.0,
                                                  rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :,
                2] = np.minimum(255.0, np.maximum(0.0,
                                                  rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,
               0:2] = transform(kp[i, 0:2] + 1,
                                center,
                                scale,
                                [self.options.IMG_RES, self.options.IMG_RES],
                                rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.IMG_RES - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()

        # EFT dataset bboxes are a bit large, make them smaller
        # if self.dataset in ['mpii', 'coco', 'lspet']:
        #     scale /= 1.1

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # apply crop augmentation
        if self.is_train and self.options.CROP_FACTOR > 0:
            if np.random.rand() < self.options.CROP_PROB:
                center, scale = random_crop(center,
                                            scale,
                                            crop_scale_factor=1 -
                                            self.options.CROP_FACTOR,
                                            axis='y')

        load_start = time.perf_counter()
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            # cv_img = np.load(imgname)['arr_0']
            cv_img = read_img(imgname)
            # cv_img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except:
            logger.info(imgname)

        orig_shape = np.array(cv_img.shape)[:2]
        load_time = time.perf_counter() - load_start
        # print(f'loading: {time.perf_counter() - loading_start}s.')

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        if self.dataset in [
                'agora-cam-v2', 'coco-cam-camreg', 'coco-campose-camreg'
        ] and self.is_train:
            pose = torch.from_numpy(pose).float()
            pose[:3] = batch_rot2aa(
                torch.from_numpy(self.cam_rotmat[index]).unsqueeze(0)
                @ batch_rodrigues(pose[:3].unsqueeze(0)))[0]

            pose = pose.numpy()

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale, rot,
                                        flip)

        proc_start = time.perf_counter()

        # Process image
        img = self.rgb_processing(cv_img,
                                  center,
                                  sc * scale,
                                  rot,
                                  flip,
                                  pn,
                                  kp2d=keypoints,
                                  img_res=self.options.IMG_RES)

        img = torch.from_numpy(img).float()
        # print(f'crop-augment: {time.perf_counter() - crop_start}s.')
        proc_time = time.perf_counter() - proc_start

        # logger.debug(f'Mode: {self.is_train}, IMG_RES={self.options.IMG_RES}, RENDER_RES={self.options.RENDER_RES}')
        if not self.is_train and not self.options.RENDER_RES == self.options.IMG_RES:
            disp_img = self.rgb_processing(cv_img,
                                           center,
                                           sc * scale,
                                           rot,
                                           flip,
                                           pn,
                                           kp2d=keypoints,
                                           img_res=self.options.RENDER_RES)
            disp_img = torch.from_numpy(disp_img).float()
            item['disp_img'] = self.normalize_img(disp_img)

        # item['orig_img'] = torch.from_numpy(np.transpose(cv_img.astype('float32'),(2,0,1)) / 255.0)
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot,
                                                             flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        pose_conf = np.ones(item['pose'].shape[0] // 3)
        if self.options.USE_3D_CONF and self.dataset in [
                'mpii', 'coco', 'lspet'
        ]:
            # copy the confidences of 2d keypoints to 3d joints
            for src, dst in kp_utils.map_spin_joints_to_smpl():
                conf = max([keypoints[x, 2] for x in src])
                pose_conf[dst] = conf
        item['pose_conf'] = pose_conf

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()

            if self.dataset in [
                    'agora-cam-v2', 'coco-cam-camreg', 'coco-campose-camreg'
            ] and self.is_train:
                S[:, :3] = (self.cam_rotmat[index] @ S[:, :3].T).T

            item['pose_3d'] = torch.from_numpy(
                self.j3d_processing(S, rot, flip)).float()
            if self.options.USE_3D_CONF and self.dataset in [
                    'mpii', 'coco', 'lspet'
            ]:
                # copy the confidences of 2d keypoints to 3d joints
                for src, dst in kp_utils.relation_among_spin_joints():
                    if len(src) < 1:
                        conf = keypoints[dst, 2]
                    else:
                        # relations + itself
                        conf = max([keypoints[x, 2]
                                    for x in src] + [keypoints[dst, 2]])
                    # logger.debug(conf.shape)
                    # subtract 25 to use only gt joints of SPIN
                    item['pose_3d'][dst - 25, -1] = conf.astype('float')
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        if self.options.NONPARAMETRIC:
            # get the nonparametric joints to train PARE
            J_regressor_batch = self.J_regressor[None, :].expand(1, -1, -1)
            if self.dataset in ['3dpw', '3dpw-all'
                                ] and self.options.USE_GENDER:
                if self.gender[index] == 1:
                    gt_vertices = self.smpl_female(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    ).vertices
                else:
                    gt_vertices = self.smpl_male(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    ).vertices
            else:
                gt_vertices = self.smpl(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),
                ).vertices

            pose_3d = torch.matmul(J_regressor_batch, gt_vertices)
            pelvis = pose_3d[:, [0], :].clone()
            pose_3d = pose_3d[:, self.joint_mapper_h36m, :]
            pose_3d = pose_3d - pelvis
            item['pose_3d'] = pose_3d[0].float()
            item['vertices'] = gt_vertices[0].float()

        # prepare pose_3d for evaluation
        # For 3DPW get the 14 common joints from the rendered shape

        if self.is_train and self.dataset == '3dpw':
            if self.gender[index] == 1:
                j3d = self.smpl_female(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),
                ).joints[0, 25:].cpu().numpy()
            else:
                j3d = self.smpl_male(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),
                ).joints[0, 25:].cpu().numpy()
            j3d -= np.array(j3d[2] + j3d[3]) / 2.  # root center
            S = np.hstack([j3d, np.ones([24, 1])])
            item['pose_3d'] = torch.from_numpy(
                self.j3d_processing(S, rot, flip)).float()

        if not self.is_train:
            if self.dataset in EVAL_MESH_DATASETS:
                J_regressor_batch = self.J_regressor[None, :].expand(1, -1, -1)
                if self.dataset in ['3dpw', '3dpw-all'
                                    ] and self.options.USE_GENDER:
                    if self.gender[index] == 1:
                        gt_vertices = self.smpl_female(
                            global_orient=item['pose'].unsqueeze(0)[:, :3],
                            body_pose=item['pose'].unsqueeze(0)[:, 3:],
                            betas=item['betas'].unsqueeze(0),
                        ).vertices
                    else:
                        gt_vertices = self.smpl_male(
                            global_orient=item['pose'].unsqueeze(0)[:, :3],
                            body_pose=item['pose'].unsqueeze(0)[:, 3:],
                            betas=item['betas'].unsqueeze(0),
                        ).vertices
                else:
                    gt_vertices = self.smpl(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    ).vertices

                pose_3d = torch.matmul(J_regressor_batch, gt_vertices)
                pelvis = pose_3d[:, [0], :].clone()
                pose_3d = pose_3d[:, self.joint_mapper_h36m, :]
                pose_3d = pose_3d - pelvis
                item['pose_3d'] = pose_3d[0].float()
                item['vertices'] = gt_vertices[0].float()
            else:
                item['pose_3d'] = item['pose_3d'][
                    self.joint_mapper_gt, :-1].float()

        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        item['load_time'] = load_time
        item['proc_time'] = proc_time
        return item

    def __len__(self):
        return len(self.imgname)
