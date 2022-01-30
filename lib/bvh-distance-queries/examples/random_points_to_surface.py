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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import argparse

try:
    input = raw_input
except NameError:
    pass

import open3d as o3d

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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh-fn', type=str, dest='mesh_fn',
                        help='A mesh file (.obj, .ply, e.t.c.) to be checked' +
                        ' for collisions')
    parser.add_argument('--num-query-points', type=int, default=1,
                        dest='num_query_points',
                        help='Number of random query points')
    parser.add_argument('--seed', type=int, default=None,
                        help='If given then set the seed')

    args, _ = parser.parse_known_args()

    mesh_fn = args.mesh_fn
    num_query_points = args.num_query_points
    seed = args.seed

    input_mesh = Mesh(filename=mesh_fn)

    if seed is not None:
        torch.manual_seed(seed)

    logger.info(f'Number of triangles = {input_mesh.f.shape[0]}')

    v = input_mesh.v

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.f.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    min_vals, _ = torch.min(vertices, dim=0, keepdim=True)
    max_vals, _ = torch.max(vertices, dim=0, keepdim=True)

    query_points = torch.rand([1, num_query_points, 3], dtype=torch.float32,
                              device=device) * (max_vals - min_vals) + min_vals
    query_points_np = query_points.detach().cpu().numpy().squeeze(
        axis=0).astype(np.float32).reshape(num_query_points, 3)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = bvh_distance_queries.BVH()

    torch.cuda.synchronize()
    start = time.perf_counter()
    distances, closest_points, closest_faces, closest_bcs = m(
        triangles, query_points)
    torch.cuda.synchronize()
    logger.info(f'CUDA Elapsed time {time.perf_counter() - start}')
    distances = distances.detach().cpu().numpy()
    closest_points = closest_points.detach().cpu().numpy().squeeze()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(input_mesh.f.astype(np.int64))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.3, 0.3])

    query_pcl = o3d.geometry.PointCloud()
    query_pcl.points = o3d.utility.Vector3dVector(
        query_points.detach().cpu().numpy().squeeze(axis=0).reshape(-1, 3))
    query_pcl.paint_uniform_color([0.9, 0.3, 0.3])

    closest_points_pcl = o3d.geometry.PointCloud()
    closest_points_pcl.points = o3d.utility.Vector3dVector(
        closest_points.reshape(-1, 3))
    closest_points_pcl.paint_uniform_color([0.3, 0.3, 0.9])

    o3d.visualization.draw_geometries([
        mesh,
        query_pcl,
        closest_points_pcl,
    ])
