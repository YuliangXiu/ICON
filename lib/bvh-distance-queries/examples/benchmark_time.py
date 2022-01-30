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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import time

import argparse

try:
    input = raw_input
except NameError:
    pass

import open3d as o3d
import pyigl as igl
from iglhelpers import p2e, e2p

import yaml

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm

from loguru import logger

from psbody.mesh import Mesh
import bvh_distance_queries


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-fn', type=str, dest='mesh_fn',
                        help='A mesh file (.obj, .ply, e.t.c.) to be checked' +
                        ' for collisions')
    parser.add_argument('--timings-fn', type=str, dest='timings_fn',
                        default='timings.yaml',
                        help='File where the timings will be saved')
    parser.add_argument('--num-query-points', type=int, default=1,
                        nargs='+',
                        dest='num_query_points_lst',
                        help='Number of random query points')
    parser.add_argument('--timing-iters', type=int, default=1000,
                        dest='timing_iters')
    parser.add_argument('--run-igl', dest='run_igl',
                        type=lambda arg: arg.lower() in ['true'],
                        default=False)

    args, _ = parser.parse_known_args()

    mesh_fn = args.mesh_fn
    num_query_points_lst = args.num_query_points_lst
    timing_iters = args.timing_iters
    timings_fn = args.timings_fn
    run_igl = args.run_igl

    input_mesh = Mesh(filename=mesh_fn)

    torch.manual_seed(0)

    logger.info(f'Number of triangles = {input_mesh.f.shape[0]}')

    v = input_mesh.v
    v -= v.mean(keepdims=True, axis=0)

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.f.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    if osp.exists(timings_fn):
        with open(timings_fn, 'r') as f:
            output_dict = yaml.load(f)
    else:
        output_dict = {}

    for num_query_points in num_query_points_lst:
        query_points = torch.rand([1, num_query_points, 3], dtype=torch.float32,
                                  device=device) * 2 - 1
        query_points_np = query_points.detach().cpu().numpy().squeeze(
            axis=0).astype(np.float32).reshape(num_query_points, 3)

        batch_size = 1
        triangles = vertices[faces].unsqueeze(dim=0)

        m = bvh_distance_queries.BVH()

        elapsed = 0
        for n in tqdm.tqdm(range(timing_iters)):
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            start = time.perf_counter()
            distances, closest_points, closest_faces = m(
                triangles, query_points)
            torch.cuda.synchronize()
            elapsed += (time.perf_counter() - start)

        cuda_elapsed = elapsed / timing_iters
        logger.info(
            f'CUDA Points = {num_query_points}: elapsed time {cuda_elapsed}')
        distances = distances.detach().cpu().numpy()
        closest_points = closest_points.detach().cpu().numpy().squeeze()
        output_dict[num_query_points]['cuda'] = cuda_elapsed

        if run_igl:
            elapsed = 0
            for n in tqdm.tqdm(range(timing_iters)):
                sqrD = igl.eigen.MatrixXd()
                closest_faces = igl.eigen.MatrixXi()
                closest_points_eig = igl.eigen.MatrixXd()
                query_points_eigen = p2e(query_points_np)
                v_eig = p2e(v)
                f_eig = p2e(input_mesh.f.astype(np.int64))

                start = time.perf_counter()
                #  Find the closest points on the SMPL-X mesh
                igl.point_mesh_squared_distance(
                    query_points_eigen,
                    v_eig, f_eig,
                    sqrD, closest_faces, closest_points_eig)
                elapsed += (time.perf_counter() - start)

            igl_elapsed = elapsed / timing_iters
            logger.info(
                f'LibIGL Points = {num_query_points}: elapsed time {igl_elapsed}')
            output_dict[num_query_points]['igl'] = igl_elapsed

    with open(timings_fn, 'w') as f:
        yaml.dump(output_dict, f)
