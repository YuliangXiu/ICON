
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
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class NormalDataset():
    def __init__(self, cfg, split='train'):

        self.split = split
        self.root = cfg.root
        self.overfit = cfg.overfit

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        self.set_splits = self.opt.set_splits
        self.scales = self.opt.scales
        self.pifu = self.opt.pifu

        # input data types and dimensions
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]
        self.in_total = self.in_nml + ['normal_F', 'normal_B']
        self.in_total_dim = self.in_nml_dim + [3, 3]

        if self.split != 'train':
            self.rotations = range(0, 360, 120)
        else:
            self.rotations = np.arange(0, 360, 360 /
                                       self.opt.rotation_num).astype(np.int)

        self.datasets_dict = {}
        for dataset_id, dataset in enumerate(self.datasets):
            dataset_dir = osp.join(self.root, dataset, "smplx")
            self.datasets_dict[dataset] = {
                "subjects":
                np.loadtxt(osp.join(self.root, dataset, "all.txt"), dtype=str),
                "path":
                dataset_dir,
                "scale":
                self.scales[dataset_id]
            }

        self.subject_list = self.get_subject_list(split)

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0, ), (1.0, ))
        ])

    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            if self.pifu:
                txt = osp.join(self.root, dataset, f'{split}_pifu.txt')
            else:
                txt = osp.join(self.root, dataset, f'{split}.txt')

            if osp.exists(txt):
                print(f"load from {txt}")
                subject_list += sorted(np.loadtxt(txt, dtype=str).tolist())

                if self.pifu:
                    miss_pifu = sorted(
                        np.loadtxt(osp.join(self.root, dataset,
                                            "miss_pifu.txt"),
                                   dtype=str).tolist())
                    subject_list = [
                        subject for subject in subject_list
                        if subject not in miss_pifu
                    ]
                    subject_list = [
                        "renderpeople/" + subject for subject in subject_list
                    ]

            else:
                train_txt = osp.join(self.root, dataset, 'train.txt')
                val_txt = osp.join(self.root, dataset, 'val.txt')
                test_txt = osp.join(self.root, dataset, 'test.txt')

                print(
                    f"generate lists of [train, val, test] \n {train_txt} \n {val_txt} \n {test_txt} \n"
                )

                split_txt = osp.join(self.root, dataset, f'{split}.txt')

                subjects = self.datasets_dict[dataset]['subjects']
                train_split = int(len(subjects) * self.set_splits[0])
                val_split = int(
                    len(subjects) * self.set_splits[1]) + train_split

                with open(train_txt, "w") as f:
                    f.write("\n".join(dataset + "/" + item
                                      for item in subjects[:train_split]))
                with open(val_txt, "w") as f:
                    f.write("\n".join(
                        dataset + "/" + item
                        for item in subjects[train_split:val_split]))
                with open(test_txt, "w") as f:
                    f.write("\n".join(dataset + "/" + item
                                      for item in subjects[val_split:]))

                subject_list += sorted(
                    np.loadtxt(split_txt, dtype=str).tolist())

        bug_list = sorted(
            np.loadtxt(osp.join(self.root, 'bug.txt'), dtype=str).tolist())

        subject_list = [
            subject for subject in subject_list if (subject not in bug_list)
        ]

        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations)

    def __getitem__(self, index):

        # only pick the first data if overfitting
        if self.overfit:
            index = 0

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]

        # choose specific test sets
        subject = self.subject_list[mid]

        subject_render = "/".join(
            [subject.split("/")[0] + "_12views",
             subject.split("/")[1]])

        # setup paths
        data_dict = {
            'dataset':
            subject.split("/")[0],
            'subject':
            subject,
            'rotation':
            rotation,
            'image_path':
            osp.join(self.root, subject_render, 'render',
                     f'{rotation:03d}.png')
        }

        # image/normal/depth loader
        for name, channel in zip(self.in_total, self.in_total_dim):

            if name != 'image':
                data_dict.update({
                    f'{name}_path':
                    osp.join(self.root, subject_render, name,
                             f'{rotation:03d}.png')
                })
            data_dict.update({
                name:
                self.imagepath2tensor(data_dict[f'{name}_path'],
                                      channel,
                                      inv='depth_B' in name)
            })

        path_keys = [
            key for key in data_dict.keys() if '_path' in key or '_dir' in key
        ]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False):

        rgba = Image.open(path).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]

        return (image * (0.5 - inv) * 2.0).float()
