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

from __future__ import division
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] *
                                          checkpoint['batch_idx']:]
        else:
            self.dataset_perm = torch.randperm(len(self.data_source)).tolist()
            self.perm = torch.randperm(len(self.data_source)).tolist()

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


class SequentialSampler(Sampler):
    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] *
                                          checkpoint['batch_idx']:]
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return len(self.perm)


class CheckpointDataLoader(DataLoader):
    """
    Extends torch.utils.data.DataLoader to handle resuming training from an arbitrary point within an epoch.
    """
    def __init__(self,
                 dataset,
                 checkpoint=None,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=True,
                 timeout=0,
                 worker_init_fn=None):

        if shuffle:
            sampler = RandomSampler(dataset, checkpoint)
        else:
            sampler = SequentialSampler(dataset, checkpoint)
        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        super(CheckpointDataLoader, self).__init__(dataset,
                                                   sampler=sampler,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   drop_last=drop_last,
                                                   pin_memory=pin_memory,
                                                   timeout=timeout,
                                                   worker_init_fn=None)
