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

import cv2
import torch
import numpy as np
from os.path import join
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
# from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

from ..models import SMPL
from ..core import constants, config
from ..core.config import DATASET_FILES, DATASET_FOLDERS
from ..utils.image_utils import crop, read_img
from .coco_occlusion import load_coco_occluders, load_pascal_occluders

# from ..utils.image_utils import crop_cv2 as crop


class SaveDataset(Dataset):
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
        super(SaveDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                       std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        self.dataset_dict = {dataset: 0}

        # disable some of the augmentations
        self.options.FLIP_PROB = 0.0
        self.options.ROT_FACTOR = 0

        if num_images > 0:
            # select a random subset of the dataset
            rand = np.random.randint(0, len(self.imgname), size=(num_images))
            logger.info(
                f'{rand.shape[0]} images are randomly sampled from {self.dataset}'
            )
            self.imgname = self.imgname[rand]
            self.data_subset = {}
            for f in self.data.files:
                self.data_subset[f] = self.data[f][rand]
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

        if 'focal_length' in self.data.files:
            self.focal_length = self.data['focal_length']

        if 'cam_rotmat' in self.data.files:
            self.cam_rotmat = self.data['cam_rotmat']

        if 'cam_pitch' in self.data.files:
            self.cam_pitch = self.data['cam_pitch']

        if 'cam_roll' in self.data.files:
            self.cam_roll = self.data['cam_roll']

        if 'cam_ext' in self.data.files:
            self.cam_ext = self.data['cam_ext']

        if 'cam_int' in self.data.files:
            self.cam_int = self.data['cam_int']

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
        if not self.is_train:
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

    def __getitem__(self, index):
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except:
            logger.info(imgname)

        # logger.debug(f'{cv_img.shape}, {imgname}')
        rgb_img = crop(cv_img,
                       center,
                       scale,
                       [int(round(200. * scale)),
                        int(round(200. * scale))],
                       rot=0)
        # logger.debug(f'{rgb_img.shape}, {imgname}')
        rgb_img = cv2.cvtColor(rgb_img.astype(np.float32), cv2.COLOR_BGR2RGB)

        cv2.imwrite(
            f'data/dataset_folders/cropped_images/{self.dataset}/{index:06d}.jpg',
            rgb_img)

        return rgb_img

    def __len__(self):
        return len(self.imgname)
