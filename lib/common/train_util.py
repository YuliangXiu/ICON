
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

import yaml
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from ..dataset.mesh_util import *
from ..net.geometry import orthogonal
from pytorch3d.renderer.mesh import rasterize_meshes
from .render_utils import Pytorch3dRasterizer
from pytorch3d.structures import Meshes
import cv2
from PIL import Image
from tqdm import tqdm
import os
from termcolor import colored


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2], sample_tensor.shape[3])
    return sample_tensor


def gen_mesh_eval(opt, net, cuda, data, resolution=None):
    resolution = opt.resolution if resolution is None else resolution
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        verts, faces, _, _ = reconstruction_faster(net,
                                                   cuda,
                                                   calib_tensor,
                                                   resolution,
                                                   b_min,
                                                   b_max,
                                                   use_octree=False)

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces = None, None
    return verts, faces


def gen_mesh(opt, net, cuda, data, save_path, resolution=None):
    resolution = opt.resolution if resolution is None else resolution
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(),
                                     (1, 2, 0)) * 0.5 +
                        0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction_faster(net, cuda, calib_tensor,
                                                   resolution, b_min, b_max)
        verts_tensor = torch.from_numpy(
            verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = netG.index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces, color = None, None, None
    return verts, faces, color


def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(),
                                     (1, 2, 0)) * 0.5 +
                        0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction_faster(netG,
                                                   cuda,
                                                   calib_tensor,
                                                   opt.resolution,
                                                   b_min,
                                                   b_max,
                                                   use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(
            verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces, color = None, None, None
    return verts, faces, color


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


# def calc_metrics(opt, net, cuda, dataset, num_tests,
#                  resolution=128, sampled_points=1000, use_kaolin=True):
#     if num_tests > len(dataset):
#         num_tests = len(dataset)
#     with torch.no_grad():
#         chamfer_arr, p2s_arr = [], []
#         for idx in tqdm(range(num_tests)):
#             data = dataset[idx * len(dataset) // num_tests]

#             verts, faces = gen_mesh_eval(opt, net, cuda, data, resolution)
#             if verts is None:
#                 continue

#             mesh_gt = trimesh.load(data['mesh_path'])
#             mesh_gt = mesh_gt.split(only_watertight=False)
#             comp_num = [mesh.vertices.shape[0] for mesh in mesh_gt]
#             mesh_gt = mesh_gt[comp_num.index(max(comp_num))]

#             mesh_pred = trimesh.Trimesh(verts, faces)

#             gt_surface_pts, _ = trimesh.sample.sample_surface_even(
#                 mesh_gt, sampled_points)
#             pred_surface_pts, _ = trimesh.sample.sample_surface_even(
#                 mesh_pred, sampled_points)

#             if use_kaolin and has_kaolin:
#                 kal_mesh_gt = kal.rep.TriangleMesh.from_tensors(
#                         torch.tensor(mesh_gt.vertices).float().to(device=cuda),
#                         torch.tensor(mesh_gt.faces).long().to(device=cuda))
#                 kal_mesh_pred = kal.rep.TriangleMesh.from_tensors(
#                     torch.tensor(mesh_pred.vertices).float().to(device=cuda),
#                     torch.tensor(mesh_pred.faces).long().to(device=cuda))

#                 kal_distance_0 = kal.metrics.mesh.point_to_surface(
#                     torch.tensor(pred_surface_pts).float().to(device=cuda), kal_mesh_gt)
#                 kal_distance_1 = kal.metrics.mesh.point_to_surface(
#                     torch.tensor(gt_surface_pts).float().to(device=cuda), kal_mesh_pred)

#                 dist_gt_pred = torch.sqrt(kal_distance_0).cpu().numpy()
#                 dist_pred_gt = torch.sqrt(kal_distance_1).cpu().numpy()
#             else:
#                 try:
#                     _, dist_pred_gt, _ = trimesh.proximity.closest_point(mesh_pred, gt_surface_pts)
#                     _, dist_gt_pred, _ = trimesh.proximity.closest_point(mesh_gt, pred_surface_pts)
#                 except Exception as e:
#                     print (e)
#                     continue

#             chamfer_dist = 0.5 * (dist_pred_gt.mean() + dist_gt_pred.mean())
#             p2s_dist = dist_pred_gt.mean()

#             chamfer_arr.append(chamfer_dist)
#             p2s_arr.append(p2s_dist)

#     return np.average(chamfer_arr), np.average(p2s_arr)


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor,
                                                      opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor,
                                     sample_tensor,
                                     calib_tensor,
                                     labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(
        prec_arr), np.average(recall_arr)


def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(
                device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(
                    color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor,
                                     netG.get_im_feat(),
                                     color_sample_tensor,
                                     calib_tensor,
                                     labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)


# pytorch lightning training related fucntions


def query_func(opt, netG, features, points, proj_matrix=None):
    '''
        - points: size of (bz, N, 3)
        - proj_matrix: size of (bz, 4, 4)
    return: size of (bz, 1, N)
    '''
    assert len(points) == 1
    samples = points.repeat(opt.num_views, 1, 1)
    samples = samples.permute(0, 2, 1)  # [bz, 3, N]

    # view specific query
    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = torch.stack([torch.eye(4).float()], dim=0).type_as(samples)

    preds = netG.query(features=features,
                       points=samples,
                       calibs=calib_tensor,
                       regressor=netG.if_regressor)

    if type(preds) is list:
        preds = preds[0]

    return preds


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def in1d(ar1, ar2):
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1, ), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def batch_mean(res, key):
    # recursive mean for multilevel dicts
    return torch.stack([
        x[key] if isinstance(x, dict) else batch_mean(x, key) for x in res
    ]).mean()


def tf_log_convert(log_dict):
    new_log_dict = log_dict.copy()
    for k, v in log_dict.items():
        new_log_dict[k.replace("_", "/")] = v
        del new_log_dict[k]

    return new_log_dict


def bar_log_convert(log_dict, name=None, rot=None):
    from decimal import Decimal

    new_log_dict = {}

    if name is not None:
        new_log_dict['name'] = name[0]
    if rot is not None:
        new_log_dict['rot'] = rot[0]

    for k, v in log_dict.items():
        color = "yellow"
        if 'loss' in k:
            color = "red"
            k = k.replace("loss", "L")
        elif 'acc' in k:
            color = "green"
            k = k.replace("acc", "A")
        elif 'iou' in k:
            color = "green"
            k = k.replace("iou", "I")
        elif 'prec' in k:
            color = "green"
            k = k.replace("prec", "P")
        elif 'recall' in k:
            color = "green"
            k = k.replace("recall", "R")

        if 'lr' not in k:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{v:.3f}", color)
        else:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{Decimal(str(v)):.1E}",
                                                   color)

    if 'loss' in new_log_dict.keys():
        del new_log_dict['loss']

    return new_log_dict


def accumulate(outputs, rot_num, split):

    hparam_log_dict = {}

    metrics = outputs[0].keys()
    datasets = split.keys()

    for dataset in datasets:
        for metric in metrics:
            keyword = f"hparam/{dataset}-{metric}"
            if keyword not in hparam_log_dict.keys():
                hparam_log_dict[keyword] = 0
            for idx in range(split[dataset][0] * rot_num,
                             split[dataset][1] * rot_num):
                hparam_log_dict[keyword] += outputs[idx][metric]
            hparam_log_dict[keyword] /= (split[dataset][1] -
                                         split[dataset][0]) * rot_num

    print(colored(hparam_log_dict, "green"))

    return hparam_log_dict


def calc_error_N(outputs, targets):
    """calculate the error of normal (IGR)

    Args:
        outputs (torch.tensor): [B, 3, N]
        target (torch.tensor): [B, N, 3]

    # manifold loss and grad_loss in IGR paper
    grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()

    Returns:
        torch.tensor: error of valid normals on the surface
    """
    # outputs = torch.tanh(-outputs.permute(0,2,1).reshape(-1,3))
    outputs = -outputs.permute(0, 2, 1).reshape(-1, 1)
    targets = targets.reshape(-1, 3)[:, 2:3]
    with_normals = targets.sum(dim=1).abs() > 0.0

    # eikonal loss
    grad_loss = ((outputs[with_normals].norm(2, dim=-1) - 1)**2).mean()
    # normals loss
    normal_loss = (outputs - targets)[with_normals].abs().norm(2, dim=1).mean()

    return grad_loss * 0.0 + normal_loss


def calc_knn_acc(preds, carn_verts, labels, pick_num):
    """calculate knn accuracy

    Args:
        preds (torch.tensor): [B, 3, N]
        carn_verts (torch.tensor): [SMPLX_V_num, 3]
        labels (torch.tensor): [B, N_knn, N]
    """
    N_knn_full = labels.shape[1]
    preds = preds.permute(0, 2, 1).reshape(-1, 3)
    labels = labels.permute(0, 2, 1).reshape(-1, N_knn_full)  # [BxN, num_knn]
    labels = labels[:, :pick_num]

    dist = torch.cdist(preds, carn_verts, p=2)  # [BxN, SMPL_V_num]
    knn = dist.topk(k=pick_num, dim=1, largest=False)[1]  # [BxN, num_knn]
    cat_mat = torch.sort(torch.cat((knn, labels), dim=1))[0]
    bool_col = torch.zeros_like(cat_mat)[:, 0]
    for i in range(pick_num * 2 - 1):
        bool_col += cat_mat[:, i] == cat_mat[:, i + 1]
    acc = (bool_col > 0).sum() / len(bool_col)

    return acc


def calc_acc_seg(output, target, num_multiseg):
    from pytorch_lightning.metrics import Accuracy
    return Accuracy()(output.reshape(-1, num_multiseg).cpu(),
                      target.flatten().cpu())


def add_watermark(imgs, titles):

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (350, 50)
    bottomRightCornerOfText = (800, 50)
    fontScale = 1
    fontColor = (1.0, 1.0, 1.0)
    lineType = 2

    for i in range(len(imgs)):

        title = titles[i + 1]
        cv2.putText(imgs[i], title, bottomLeftCornerOfText, font, fontScale,
                    fontColor, lineType)

        if i == 0:
            cv2.putText(imgs[i], str(titles[i][0]), bottomRightCornerOfText,
                        font, fontScale, fontColor, lineType)

    result = np.concatenate(imgs, axis=0).transpose(2, 0, 1)

    return result


def make_test_gif(img_dir):

    if img_dir is not None and len(os.listdir(img_dir)) > 0:
        for dataset in os.listdir(img_dir):
            for subject in sorted(os.listdir(osp.join(img_dir, dataset))):
                img_lst = []
                im1 = None
                for file in sorted(
                        os.listdir(osp.join(img_dir, dataset, subject))):
                    if file[-3:] not in ['obj', 'gif']:
                        img_path = os.path.join(img_dir, dataset, subject,
                                                file)
                        if im1 == None:
                            im1 = Image.open(img_path)
                        else:
                            img_lst.append(Image.open(img_path))

                print(os.path.join(img_dir, dataset, subject, "out.gif"))
                im1.save(os.path.join(img_dir, dataset, subject, "out.gif"),
                         save_all=True,
                         append_images=img_lst,
                         duration=500,
                         loop=0)


def export_cfg(logger, cfg):

    cfg_export_file = osp.join(logger.save_dir, logger.name,
                               f"version_{logger.version}", "cfg.yaml")

    if not osp.exists(cfg_export_file):
        os.makedirs(osp.dirname(cfg_export_file), exist_ok=True)
        with open(cfg_export_file, "w+") as file:
            _ = yaml.dump(cfg, file)
