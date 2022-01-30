
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_mask2D(mask,
                title="",
                point_coords=None,
                figsize=10,
                point_marker_size=5):
    '''
    Simple plotting tool to show intermediate mask predictions and points 
    where PointRend is applied.

    Args:
    mask (Tensor): mask prediction of shape HxW
    title (str): title for the plot
    point_coords ((Tensor, Tensor)): x and y point coordinates
    figsize (int): size of the figure to plot
    point_marker_size (int): marker size for points
    '''

    H, W = mask.shape
    plt.figure(figsize=(figsize, figsize))
    if title:
        title += ", "
    plt.title("{}resolution {}x{}".format(title, H, W), fontsize=30)
    plt.ylabel(H, fontsize=30)
    plt.xlabel(W, fontsize=30)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(mask.detach(),
               interpolation="nearest",
               cmap=plt.get_cmap('gray'))
    if point_coords is not None:
        plt.scatter(x=point_coords[0],
                    y=point_coords[1],
                    color="red",
                    s=point_marker_size,
                    clip_on=True)
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    plt.show()


def plot_mask3D(mask=None,
                title="",
                point_coords=None,
                figsize=1500,
                point_marker_size=8,
                interactive=True):
    '''
    Simple plotting tool to show intermediate mask predictions and points 
    where PointRend is applied.

    Args:
    mask (Tensor): mask prediction of shape DxHxW
    title (str): title for the plot
    point_coords ((Tensor, Tensor, Tensor)): x and y and z point coordinates
    figsize (int): size of the figure to plot
    point_marker_size (int): marker size for points
    '''
    import trimesh
    import vtkplotter
    from skimage import measure

    vp = vtkplotter.Plotter(title=title, size=(figsize, figsize))
    vis_list = []

    if mask is not None:
        mask = mask.detach().to("cpu").numpy()
        mask = mask.transpose(2, 1, 0)

        # marching cube to find surface
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            mask, 0.5, gradient_direction='ascent')

        # create a mesh
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        vis_list.append(mesh)

    if point_coords is not None:
        point_coords = torch.stack(point_coords, 1).to("cpu").numpy()

        # import numpy as np
        # select_x = np.logical_and(point_coords[:, 0] >= 16, point_coords[:, 0] <= 112)
        # select_y = np.logical_and(point_coords[:, 1] >= 48, point_coords[:, 1] <= 272)
        # select_z = np.logical_and(point_coords[:, 2] >= 16, point_coords[:, 2] <= 112)
        # select = np.logical_and(np.logical_and(select_x, select_y), select_z)
        # point_coords = point_coords[select, :]

        pc = vtkplotter.Points(point_coords, r=point_marker_size, c='red')
        vis_list.append(pc)

    vp.show(*vis_list,
            bg="white",
            axes=1,
            interactive=interactive,
            azimuth=30,
            elevation=30)


def create_grid3D(min, max, steps):
    if type(min) is int:
        min = (min, min, min)  # (x, y, z)
    if type(max) is int:
        max = (max, max, max)  # (x, y)
    if type(steps) is int:
        steps = (steps, steps, steps)  # (x, y, z)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).long()
    gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH,
                          gridD])  # [2, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t()  # [N, 3]
    return coords


def create_grid2D(min, max, steps):
    if type(min) is int:
        min = (min, min)  # (x, y)
    if type(max) is int:
        max = (max, max)  # (x, y)
    if type(steps) is int:
        steps = (steps, steps)  # (x, y)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    girdH, gridW = torch.meshgrid([arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH])  # [2, steps[0], steps[1]]
    coords = coords.view(2, -1).t()  # [N, 2]
    return coords


class SmoothConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2

        weight = torch.ones(
            (in_channels, out_channels, kernel_size, kernel_size),
            dtype=torch.float32) / (kernel_size**2)
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=self.padding)


class SmoothConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2

        weight = torch.ones(
            (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
            dtype=torch.float32) / (kernel_size**3)
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv3d(input, self.weight, padding=self.padding)


def build_smooth_conv3D(in_channels=1,
                        out_channels=1,
                        kernel_size=3,
                        padding=1):
    smooth_conv = torch.nn.Conv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding)
    smooth_conv.weight.data = torch.ones(
        (in_channels, out_channels, kernel_size, kernel_size, kernel_size),
        dtype=torch.float32) / (kernel_size**3)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv


def build_smooth_conv2D(in_channels=1,
                        out_channels=1,
                        kernel_size=3,
                        padding=1):
    smooth_conv = torch.nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding)
    smooth_conv.weight.data = torch.ones(
        (in_channels, out_channels, kernel_size, kernel_size),
        dtype=torch.float32) / (kernel_size**2)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv


def get_uncertain_point_coords_on_grid3D(uncertainty_map, num_points,
                                         **kwargs):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W, D) that contains uncertainty
            values for a set of points on a regular H x W x D grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W x D) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W x D grid.
    """
    R, _, D, H, W = uncertainty_map.shape
    # h_step = 1.0 / float(H)
    # w_step = 1.0 / float(W)
    # d_step = 1.0 / float(D)

    num_points = min(D * H * W, num_points)
    point_scores, point_indices = torch.topk(uncertainty_map.view(
        R, D * H * W),
                                             k=num_points,
                                             dim=1)
    point_coords = torch.zeros(R,
                               num_points,
                               3,
                               dtype=torch.float,
                               device=uncertainty_map.device)
    # point_coords[:, :, 0] = h_step / 2.0 + (point_indices // (W * D)).to(torch.float) * h_step
    # point_coords[:, :, 1] = w_step / 2.0 + (point_indices % (W * D) // D).to(torch.float) * w_step
    # point_coords[:, :, 2] = d_step / 2.0 + (point_indices % D).to(torch.float) * d_step
    point_coords[:, :, 0] = (point_indices % W).to(torch.float)  # x
    point_coords[:, :, 1] = (point_indices % (H * W) // W).to(torch.float)  # y
    point_coords[:, :, 2] = (point_indices // (H * W)).to(torch.float)  # z
    print(f"resolution {D} x {H} x {W}", point_scores.min(),
          point_scores.max())
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid3D_faster(uncertainty_map, num_points,
                                                clip_min):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W, D) that contains uncertainty
            values for a set of points on a regular H x W x D grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W x D) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W x D grid.
    """
    R, _, D, H, W = uncertainty_map.shape
    # h_step = 1.0 / float(H)
    # w_step = 1.0 / float(W)
    # d_step = 1.0 / float(D)

    assert R == 1, "batchsize > 1 is not implemented!"
    uncertainty_map = uncertainty_map.view(D * H * W)
    indices = (uncertainty_map >= clip_min).nonzero().squeeze(1)
    num_points = min(num_points, indices.size(0))
    point_scores, point_indices = torch.topk(uncertainty_map[indices],
                                             k=num_points,
                                             dim=0)
    point_indices = indices[point_indices].unsqueeze(0)

    point_coords = torch.zeros(R,
                               num_points,
                               3,
                               dtype=torch.float,
                               device=uncertainty_map.device)
    # point_coords[:, :, 0] = h_step / 2.0 + (point_indices // (W * D)).to(torch.float) * h_step
    # point_coords[:, :, 1] = w_step / 2.0 + (point_indices % (W * D) // D).to(torch.float) * w_step
    # point_coords[:, :, 2] = d_step / 2.0 + (point_indices % D).to(torch.float) * d_step
    point_coords[:, :, 0] = (point_indices % W).to(torch.float)  # x
    point_coords[:, :, 1] = (point_indices % (H * W) // W).to(torch.float)  # y
    point_coords[:, :, 2] = (point_indices // (H * W)).to(torch.float)  # z
    # print (f"resolution {D} x {H} x {W}", point_scores.min(), point_scores.max())
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid2D(uncertainty_map, num_points,
                                         **kwargs):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    # h_step = 1.0 / float(H)
    # w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_scores, point_indices = torch.topk(uncertainty_map.view(R, H * W),
                                             k=num_points,
                                             dim=1)
    point_coords = torch.zeros(R,
                               num_points,
                               2,
                               dtype=torch.long,
                               device=uncertainty_map.device)
    # point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    # point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    point_coords[:, :, 0] = (point_indices % W).to(torch.long)
    point_coords[:, :, 1] = (point_indices // W).to(torch.long)
    # print (point_scores.min(), point_scores.max())
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid2D_faster(uncertainty_map, num_points,
                                                clip_min):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    # h_step = 1.0 / float(H)
    # w_step = 1.0 / float(W)

    assert R == 1, "batchsize > 1 is not implemented!"
    uncertainty_map = uncertainty_map.view(H * W)
    indices = (uncertainty_map >= clip_min).nonzero().squeeze(1)
    num_points = min(num_points, indices.size(0))
    point_scores, point_indices = torch.topk(uncertainty_map[indices],
                                             k=num_points,
                                             dim=0)
    point_indices = indices[point_indices].unsqueeze(0)

    point_coords = torch.zeros(R,
                               num_points,
                               2,
                               dtype=torch.long,
                               device=uncertainty_map.device)
    # point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    # point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    point_coords[:, :, 0] = (point_indices % W).to(torch.long)
    point_coords[:, :, 1] = (point_indices // W).to(torch.long)
    # print (point_scores.min(), point_scores.max())
    return point_indices, point_coords


def calculate_uncertainty(logits, classes=None, balance_value=0.5):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device),
            classes].unsqueeze(1)
    return -torch.abs(gt_class_logits - balance_value)


def grid_interp(vol, points):
    """
    Interpolate volume data at given points
    Inputs:
        vol: 4D torch tensor (C, Nz, Ny, Nx)
        points: point locations (Np, 3)
    Outputs:
        output: interpolated data (Np, C)    
    """

    if vol.is_cuda:
        return mc.grid_interp_cuda(vol, points)
    else:
        return mc.grid_interp_cpu(vol, points)
