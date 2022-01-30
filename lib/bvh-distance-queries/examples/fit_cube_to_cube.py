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
# Author: Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de
# Contact: ps-license@tuebingen.mpg.de

import sys
import os

import time

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

import numpy as np

from loguru import logger

from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
import kornia

import bvh_distance_queries


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pause', type=float, default=None,
                        help='Pause duration for the viewer')

    args, _ = parser.parse_known_args()
    pause = args.pause

    batch_size = 1
    m = bvh_distance_queries.PointToMeshResidual()

    template = Mesh(filename='data/test_box.ply')

    template_v = torch.tensor(
        template.v, dtype=torch.float32, device=device).reshape(1, -1, 3)
    template_f = torch.tensor(
        template.f.astype(np.int64),
        dtype=torch.long, device=device).reshape(-1, 3)

    template_translation = torch.tensor(
        [3, 2, 1], dtype=torch.float32,
        device=device).reshape(1, 3)
    template_rotation = torch.tensor([1, 2, 3], dtype=torch.float32,
                                     device=device).reshape(1, 3)
    template_rotation.requires_grad_(True)
    template_translation.requires_grad_(True)

    scan_points = torch.tensor(
        template.v, dtype=torch.float32, device=device).reshape(1, -1, 3)

    optimizer = optim.LBFGS([template_translation, template_rotation],
                            lr=1, line_search_fn='strong_wolfe',
                            max_iter=20)

    scan = deepcopy(template)
    scan.vc = np.ones_like(scan.v) * [0.3, 0.3, 0.3]

    mv = MeshViewer()

    def closure(visualize=False, backward=True):
        if backward:
            optimizer.zero_grad()

        rot_mat = kornia.angle_axis_to_rotation_matrix(template_rotation)

        vertices = torch.einsum(
            'bij,bmj->bmi',
            [rot_mat, template_v]) + template_translation.unsqueeze(dim=1)

        triangles = vertices[:, template_f].contiguous()

        residual, _ = m(triangles, scan_points)
        loss = residual.pow(2).sum(dim=-1).mean()

        if backward:
            loss.backward()

        if visualize:
            template.v = vertices.detach().cpu().numpy().squeeze()
            mv.set_static_meshes([template, scan])
            if pause is not None:
                time.sleep(pause)
            else:
                logger.info('Press escape to exit ...')
                logger.info('Waiting for key ...')
                key = mv.get_keypress()
                if key == b'\x1b':
                    logger.warning('Exiting!')
                    sys.exit(0)

        return loss

    closure(visualize=True, backward=False)
    N = 1000
    for n in range(N):
        curr_loss = optimizer.step(closure)
        closure(visualize=True, backward=False)

        verts_dist = np.sqrt(np.power(
            scan_points.detach().cpu().numpy().squeeze() - template.v,
            2).sum(axis=-1)).mean()
        logger.info(f'[{n:03d}]: {curr_loss.item():.4f}')
        logger.info(f'[{n:03d}]: Vertex-to-vertex distance: {verts_dist} (m)')
