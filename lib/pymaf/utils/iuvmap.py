import torch
import torch.nn.functional as F


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

        for part_id in range(0, K):
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

            # x1 = roi_fg[0]
            # x2 = roi_fg[2]
            # y1 = roi_fg[1]
            # y2 = roi_fg[3]

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


def iuv_img2map(uvimages, uv_rois=None, new_size=None):
    device_id = uvimages.get_device()
    batch_size = uvimages.size(0)
    uvimg_size = uvimages.size(-1)

    Index2mask = [[0], [1, 2], [3], [4], [5], [6], [7, 9], [8, 10], [11, 13],
                  [12, 14], [15, 17], [16, 18], [19, 21], [20, 22], [23, 24]]

    part_ind = torch.round(uvimages[:, 0, :, :] * 24)
    part_u = uvimages[:, 1, :, :]
    part_v = uvimages[:, 2, :, :]

    recon_U = []
    recon_V = []
    recon_Index_UV = []
    recon_Ann_Index = []

    for i in range(25):
        if i == 0:
            recon_Index_UV_i = torch.min(F.threshold(part_ind + 1, 0.5, 0),
                                         -F.threshold(-part_ind - 1, -1.5, 0))
        else:
            recon_Index_UV_i = torch.min(
                F.threshold(part_ind, i - 0.5, 0),
                -F.threshold(-part_ind, -i - 0.5, 0)) / float(i)
        recon_U_i = recon_Index_UV_i * part_u
        recon_V_i = recon_Index_UV_i * part_v

        recon_Index_UV.append(recon_Index_UV_i)
        recon_U.append(recon_U_i)
        recon_V.append(recon_V_i)

    for i in range(len(Index2mask)):
        if len(Index2mask[i]) == 1:
            recon_Ann_Index_i = recon_Index_UV[Index2mask[i][0]]
        elif len(Index2mask[i]) == 2:
            p_ind0 = Index2mask[i][0]
            p_ind1 = Index2mask[i][1]
            # recon_Ann_Index[:, i, :, :] = torch.where(recon_Index_UV[:, p_ind0, :, :] > 0.5, recon_Index_UV[:, p_ind0, :, :], recon_Index_UV[:, p_ind1, :, :])
            # recon_Ann_Index[:, i, :, :] = torch.eq(part_ind, p_ind0) | torch.eq(part_ind, p_ind1)
            recon_Ann_Index_i = recon_Index_UV[p_ind0] + recon_Index_UV[p_ind1]

        recon_Ann_Index.append(recon_Ann_Index_i)

    recon_U = torch.stack(recon_U, dim=1)
    recon_V = torch.stack(recon_V, dim=1)
    recon_Index_UV = torch.stack(recon_Index_UV, dim=1)
    recon_Ann_Index = torch.stack(recon_Ann_Index, dim=1)

    if uv_rois is None:
        return recon_U, recon_V, recon_Index_UV, recon_Ann_Index

    recon_U_roi = []
    recon_V_roi = []
    recon_Index_UV_roi = []
    recon_Ann_Index_roi = []

    if new_size is None:
        M = uvimg_size
    else:
        M = new_size

    for i in range(batch_size):
        roi_fg = uv_rois[i][1:]

        # x1 = roi_fg[0]
        # x2 = roi_fg[2]
        # y1 = roi_fg[1]
        # y2 = roi_fg[3]

        w = roi_fg[2] - roi_fg[0]
        h = roi_fg[3] - roi_fg[1]

        aspect_ratio = float(w) / h

        if aspect_ratio < 1:
            w_size = max(int(uvimg_size * aspect_ratio), 1)
            w_margin = int((uvimg_size - w_size) / 2)

            recon_U_roi_i = recon_U[i, :, :, w_margin:w_margin + w_size]
            recon_V_roi_i = recon_V[i, :, :, w_margin:w_margin + w_size]
            recon_Index_UV_roi_i = recon_Index_UV[i, :, :,
                                                  w_margin:w_margin + w_size]
            recon_Ann_Index_roi_i = recon_Ann_Index[i, :, :,
                                                    w_margin:w_margin + w_size]
        else:
            h_size = max(int(uvimg_size / aspect_ratio), 1)
            h_margin = int((uvimg_size - h_size) / 2)

            recon_U_roi_i = recon_U[i, :, h_margin:h_margin + h_size, :]
            recon_V_roi_i = recon_V[i, :, h_margin:h_margin + h_size, :]
            recon_Index_UV_roi_i = recon_Index_UV[i, :, h_margin:h_margin +
                                                  h_size, :]
            recon_Ann_Index_roi_i = recon_Ann_Index[i, :, h_margin:h_margin +
                                                    h_size, :]

        recon_U_roi_i = F.interpolate(recon_U_roi_i.unsqueeze(0),
                                      size=(M, M),
                                      mode='nearest')
        recon_V_roi_i = F.interpolate(recon_V_roi_i.unsqueeze(0),
                                      size=(M, M),
                                      mode='nearest')
        recon_Index_UV_roi_i = F.interpolate(recon_Index_UV_roi_i.unsqueeze(0),
                                             size=(M, M),
                                             mode='nearest')
        recon_Ann_Index_roi_i = F.interpolate(
            recon_Ann_Index_roi_i.unsqueeze(0), size=(M, M), mode='nearest')

        recon_U_roi.append(recon_U_roi_i)
        recon_V_roi.append(recon_V_roi_i)
        recon_Index_UV_roi.append(recon_Index_UV_roi_i)
        recon_Ann_Index_roi.append(recon_Ann_Index_roi_i)

    recon_U_roi = torch.cat(recon_U_roi, dim=0)
    recon_V_roi = torch.cat(recon_V_roi, dim=0)
    recon_Index_UV_roi = torch.cat(recon_Index_UV_roi, dim=0)
    recon_Ann_Index_roi = torch.cat(recon_Ann_Index_roi, dim=0)

    return recon_U_roi, recon_V_roi, recon_Index_UV_roi, recon_Ann_Index_roi
