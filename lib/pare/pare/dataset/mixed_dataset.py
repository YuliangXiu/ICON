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
from .base_dataset import BaseDataset
from .occlusion import load_occluders


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        self.dataset_list = [
            'h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp'
        ]
        self.dataset_dict = {
            'h36m': 0,
            'lsp-orig': 1,
            'mpii': 2,
            'lspet': 3,
            'coco': 4,
            'mpi-inf-3dhp': 5
        }

        occluders = None
        if options.USE_SYNTHETIC_OCCLUSION:
            logger.info('Loading synthetic occluders for mixed dataset.')
            occluders = load_occluders(pascal_voc_root_path=config.PASCAL_ROOT)
            logger.info('Found {} suitable objects'.format(len(occluders)))

        self.datasets = [
            BaseDataset(options, ds, occluders=occluders, **kwargs)
            for ds in self.dataset_list
        ]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = [
            .3, .6 * len(self.datasets[1]) / length_itw,
            .6 * len(self.datasets[2]) / length_itw,
            .6 * len(self.datasets[3]) / length_itw,
            .6 * len(self.datasets[4]) / length_itw, 0.1
        ]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
