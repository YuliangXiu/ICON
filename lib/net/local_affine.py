# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import torch.nn as nn
import torch.sparse as sp

# reference: https://github.com/wuhaozhe/pytorch-nicp
class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size=1, edges=None):
        '''
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        '''
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(torch.eye(3).unsqueeze(
            0).unsqueeze(0).repeat(batch_size, num_points, 1, 1))
        self.b = nn.Parameter(torch.zeros(3).unsqueeze(0).unsqueeze(
            0).unsqueeze(3).repeat(batch_size, num_points, 1, 1))
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        '''
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix, 
        '''
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2) ** 2
        w_rigid = (torch.linalg.det(self.A) - 1.0) ** 2
        return w_diff, w_rigid

    def forward(self, x, return_stiff=False):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness, rigid = self.stiffness()
            return out_x, stiffness, rigid
        else:
            return out_x
