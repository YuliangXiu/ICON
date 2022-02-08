
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

import sys
import numpy as np
from torch.utils.data import DataLoader
from .NormalDataset import NormalDataset

# pytorch lightning related libs
import pytorch_lightning as pl


class NormalModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(NormalModule, self).__init__()
        self.cfg = cfg
        self.overfit = self.cfg.overfit

        if self.overfit:
            self.batch_size = 1
        else:
            self.batch_size = self.cfg.batch_size

        self.data_size = {}

    def prepare_data(self):

        pass

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def setup(self, stage):

        if stage == 'fit' or stage is None:
            self.train_dataset = NormalDataset(cfg=self.cfg, split="train")
            self.val_dataset = NormalDataset(cfg=self.cfg, split="val")
            self.data_size = {
                'train': len(self.train_dataset),
                'val': len(self.val_dataset)
            }

        if stage == 'test' or stage is None:
            self.test_dataset = NormalDataset(cfg=self.cfg, split="test")

    def train_dataloader(self):

        train_data_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=not self.overfit,
                                       num_workers=self.cfg.num_threads,
                                       pin_memory=True,
                                       worker_init_fn=self.worker_init_fn)

        return train_data_loader

    def val_dataloader(self):

        if self.overfit:
            current_dataset = self.train_dataset
        else:
            current_dataset = self.val_dataset

        val_data_loader = DataLoader(current_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.cfg.num_threads,
                                     pin_memory=True)

        return val_data_loader

    def test_dataloader(self):

        test_data_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=self.cfg.num_threads,
                                      pin_memory=True)

        return test_data_loader
