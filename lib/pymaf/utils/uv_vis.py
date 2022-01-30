import os
import torch
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
# Use a non-interactive backend
import matplotlib

matplotlib.use('Agg')

from .renderer import OpenDRenderer


def iuv_map2img(U_uv,
                V_uv,
                Index_UV,
                AnnIndex=None,
                uv_rois=None,
                ind_mapping=None):
    device_id = U_uv.get_device()
    batch_size = U_uv.size(0)
    K = U_uv.size(1)
    heatmap_size = U_uv.size(2)

    Index_UV_max = torch.argmax(Index_UV, dim=1)
    if AnnIndex is None:
        Index_UV_max = Index_UV_max.to(torch.int64)
    else:
        AnnIndex_max = torch.argmax(AnnIndex, dim=1)
        Index_UV_max = Index_UV_max * (AnnIndex_max > 0).to(torch.int64)

    outputs = []

    for batch_id in range(batch_size):
        output = torch.zeros([3, U_uv.size(2), U_uv.size(3)],
                             dtype=torch.float32).cuda(device_id)
        output[0] = Index_UV_max[batch_id].to(torch.float32)
        if ind_mapping is None:
            output[0] /= float(K - 1)
        else:
            for ind in range(len(ind_mapping)):
                output[0][output[0] == ind] = ind_mapping[ind] * (1. / 24.)

        for part_id in range(1, K):
            CurrentU = U_uv[batch_id, part_id]
            CurrentV = V_uv[batch_id, part_id]
            output[1, Index_UV_max[batch_id] == part_id] = CurrentU[
                Index_UV_max[batch_id] == part_id]
            output[2, Index_UV_max[batch_id] == part_id] = CurrentV[
                Index_UV_max[batch_id] == part_id]

        if uv_rois is None:
            outputs.append(output.unsqueeze(0))
        else:
            roi_fg = uv_rois[batch_id][1:]
            w = roi_fg[2] - roi_fg[0]
            h = roi_fg[3] - roi_fg[1]

            aspect_ratio = float(w) / h

            if aspect_ratio < 1:
                new_size = [
                    heatmap_size,
                    max(int(heatmap_size * aspect_ratio), 1)
                ]
                output = F.interpolate(output.unsqueeze(0),
                                       size=new_size,
                                       mode='nearest')
                paddingleft = int(0.5 * (heatmap_size - new_size[1]))
                output = F.pad(output,
                               pad=(paddingleft,
                                    heatmap_size - new_size[1] - paddingleft,
                                    0, 0))
            else:
                new_size = [
                    max(int(heatmap_size / aspect_ratio), 1), heatmap_size
                ]
                output = F.interpolate(output.unsqueeze(0),
                                       size=new_size,
                                       mode='nearest')
                paddingtop = int(0.5 * (heatmap_size - new_size[0]))
                output = F.pad(output,
                               pad=(0, 0, paddingtop,
                                    heatmap_size - new_size[0] - paddingtop))

            outputs.append(output)

    return torch.cat(outputs, dim=0)


def vis_smpl_iuv(image, cam_pred, vert_pred, face, pred_uv, vert_errors_batch,
                 image_name, save_path, opt):

    # save_path = os.path.join('./notebooks/output/demo_results-wild', ids[f_id][0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dr_render = OpenDRenderer(ratio=opt.ratio)

    focal_length = 5000.
    orig_size = 224.

    if pred_uv is not None:
        iuv_img = iuv_map2img(*pred_uv)

    for draw_i in range(len(cam_pred)):
        err_val = '{:06d}_'.format(int(10 * vert_errors_batch[draw_i]))
        draw_name = err_val + image_name[draw_i]
        K = np.array([[focal_length, 0., orig_size / 2.],
                      [0., focal_length, orig_size / 2.], [0., 0., 1.]])

        img_orig, img_resized, img_smpl, render_smpl_rgba = dr_render(
            image[draw_i], cam_pred[draw_i], K, vert_pred[draw_i], face,
            draw_name[:-4])

        ones_img = np.ones(img_smpl.shape[:2]) * 255
        ones_img = ones_img[:, :, None]
        img_smpl_rgba = np.concatenate((img_smpl * 255, ones_img), axis=2)
        img_resized_rgba = np.concatenate((img_resized * 255, ones_img),
                                          axis=2)

        render_img = np.concatenate(
            (img_resized_rgba, img_smpl_rgba, render_smpl_rgba * 255), axis=1)
        render_img[render_img < 0] = 0
        render_img[render_img > 255] = 255
        matplotlib.image.imsave(
            os.path.join(save_path, draw_name[:-4] + '.png'),
            render_img.astype(np.uint8))

        if pred_uv is not None:
            # estimated global IUV
            global_iuv = iuv_img[draw_i].cpu().numpy()
            global_iuv = np.transpose(global_iuv, (1, 2, 0))
            global_iuv = resize(global_iuv, img_resized.shape[:2])
            global_iuv[global_iuv > 1] = 1
            global_iuv[global_iuv < 0] = 0
            matplotlib.image.imsave(
                os.path.join(save_path, 'pred_uv_' + draw_name[:-4] + '.png'),
                global_iuv)
