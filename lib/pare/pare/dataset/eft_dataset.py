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
"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
from loguru import logger

from ..core import config
from ..utils.train_utils import parse_datasets_ratios
from .coco_occlusion import load_coco_occluders, load_pascal_occluders


class EFTDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        datasets_ratios = parse_datasets_ratios(options.DATASETS_AND_RATIOS)
        hl = len(datasets_ratios) // 2
        self.dataset_list = datasets_ratios[:hl]
        self.dataset_ratios = datasets_ratios[hl:]

        assert len(self.dataset_list) == len(
            self.dataset_ratios
        ), 'Number of datasets and ratios should be equal'

        # self.dataset_list = ['h36m', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        self.dataset_dict = {
            'h36m': 0,
            'mpii': 1,
            'lspet': 2,
            'coco': 3,
            'mpi-inf-3dhp': 4,
            '3doh': 5
        }
        itw_datasets = ['mpii', 'lspet', 'coco']
        occluders = None
        if options.USE_SYNTHETIC_OCCLUSION:
            logger.info('Loading synthetic occluders for eft dataset.')
            # occluders = load_occluders(pascal_voc_root_path=config.PASCAL_ROOT)
            # logger.info('Found {} suitable objects'.format(len(occluders)))

            if options.OCC_AUG_DATASET == 'coco':
                occluders = load_coco_occluders()
                logger.info(f'Found {len(occluders["obj_class"])} suitable '
                            f'objects from {options.OCC_AUG_DATASET} dataset')
            elif options.OCC_AUG_DATASET == 'pascal':
                occluders = load_pascal_occluders(
                    pascal_voc_root_path=config.PASCAL_ROOT)
                logger.info(f'Found {len(occluders)} suitable '
                            f'objects from {options.OCC_AUG_DATASET} dataset')

        self.datasets = [
            eval(f'{options.LOAD_TYPE}Dataset')(options,
                                                ds,
                                                occluders=occluders,
                                                **kwargs)
            for ds in self.dataset_list
        ]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum(
            [len(ds) for ds in self.datasets if ds.dataset in itw_datasets])
        self.length = max([len(ds) for ds in self.datasets])

        self.partition = []

        for idx, (ds_name, ds_ratio) in enumerate(
                zip(self.dataset_list, self.dataset_ratios)):
            if ds_name in itw_datasets:
                r = ds_ratio * len(self.datasets[idx]) / length_itw
            else:
                r = ds_ratio

            self.partition.append(r)

        # self.partition = [
        #     .3,
        #     .6*len(self.datasets[1])/length_itw,
        #     .6*len(self.datasets[2])/length_itw,
        #     .6*len(self.datasets[3])/length_itw,
        #     0.1
        # ]
        logger.info(f'Using these datasets: {self.dataset_list}')
        logger.info(f'Ratios of datasets: {self.partition}')

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
