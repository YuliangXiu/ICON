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
# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de
# Contact: ps-license@tuebingen.mpg.de

from typing import Tuple, NewType

import torch
import torch.nn as nn
import torch.autograd as autograd
from trimesh.proximity import closest_point

from .bvh_search_tree import BVH

Tensor = NewType('Tensor', torch.Tensor)


class PointToMeshResidual(nn.Module):

    def __init__(self,
                 sort_points_by_morton: bool = True,
                 queue_size: int = 128) -> None:
        ''' Constructor for the point to mesh residual module

            Parameters
            ----------
                sort_points_by_morton: bool, optional
                    Sort input points by their morton code. Helps improve query
                    speed. Default is true
                queue_size: int, optional
                    The size of the data structure used to store intermediate
                    distance computations
        '''
        super(PointToMeshResidual, self).__init__()
        self.search_tree = BVH(sort_points_by_morton=sort_points_by_morton,
                               queue_size=queue_size)

    def forward(self,
                triangles: Tensor,
                points: Tensor,
                normals: Tensor,
                cmaps: Tensor,
                faces: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ''' Forward pass of the search tree

            Parameters
            ----------
                triangles: torch.tensor
                    A BxFx3x3 PyTorch tensor that contains the triangle
                    locations.
                points: torch.tensor
                    A BxQx3 PyTorch tensor that contains the query point
                    locations.
                normals: torch.tensor
                    A BxFx3x3 PyTorch tensor that contains the normal
                    vectors.
                faces: torch.tensor
                    A BxFx3 PyTorch tensor that contains the face
                    vectors
            Returns
            -------
                residuals: torch.tensor
                    A BxQx3 tensor with the vector that points from the query
                    to the closest point
        '''
        output = self.search_tree(triangles, points)
        distances, _, closest_faces, closest_bcs = output

        closest_bcs = torch.clamp(closest_bcs, 0, 1)

        batch_size, num_triangles = triangles.shape[:2]
        num_points = points.shape[1]

        closest_faces_idxs = (
            torch.arange(
                0, batch_size, device=triangles.device, dtype=torch.long) *
            num_triangles
        ).view(batch_size, 1)
        
        def compute_bcs_result(features):

            closest_triangles = features.view(-1, 3, 3)[
                closest_faces_idxs + closest_faces].view(
                    batch_size, num_points, 3, 3)
                
            closest_features = (
                closest_triangles[:, :, 0] *
                closest_bcs[:, :, 0].unsqueeze(dim=-1) +
                closest_triangles[:, :, 1] *
                closest_bcs[:, :, 1].unsqueeze(dim=-1) +
                closest_triangles[:, :, 2] *
                closest_bcs[:, :, 2].unsqueeze(dim=-1)
            )
            
            return closest_features
            
        closest_points = compute_bcs_result(triangles)
        closest_normals = compute_bcs_result(normals)
        closest_cmaps = compute_bcs_result(cmaps)
            
        residual = closest_points - points
        
        # faces [B, FN, 3]
        # cloesest_faces [B, CN]
        # cloesest_bcs [B, CN, 3]
        closest_fids = torch.gather(faces, 1, torch.tile(closest_faces.unsqueeze(-1),(1,1,3)))
        closest_idx = torch.gather(closest_fids, 2, closest_bcs.max(2)[1].unsqueeze(-1)).squeeze(-1)
        
        return residual, closest_normals, closest_cmaps, closest_idx
