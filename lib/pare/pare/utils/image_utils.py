"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import torch
import joblib
from skimage.transform import rotate, resize
import numpy as np
import jpeg4py as jpeg
from trimesh.visual import color

from ..core import constants
from .vibe_image_utils import gen_trans_from_patch_cv
from .kp_utils import map_smpl_to_common


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot)  # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # resize image
    new_img = resize(new_img, res)  # scipy.misc.imresize(new_img, res)
    return new_img


def crop_cv2(img, center, scale, res, rot=0):
    c_x, c_y = center
    c_x, c_y = int(round(c_x)), int(round(c_y))
    patch_width, patch_height = int(round(res[0])), int(round(res[1]))
    bb_width = bb_height = int(round(scale * 200.))

    trans = gen_trans_from_patch_cv(
        c_x,
        c_y,
        bb_width,
        bb_height,
        patch_width,
        patch_height,
        scale=1.0,
        rot=rot,
        inv=False,
    )

    crop_img = cv2.warpAffine(img,
                              trans, (int(patch_width), int(patch_height)),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    return crop_img


def get_random_crop_coords(height, width, crop_height, crop_width, h_start,
                           w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(center, scale, crop_scale_factor, axis='all'):
    '''
    center: bbox center [x,y]
    scale: bbox height / 200
    crop_scale_factor: amount of cropping to be applied
    axis: axis which cropping will be applied
        "x": center the y axis and get random crops in x
        "y": center the x axis and get random crops in y
        "all": randomly crop from all locations
    '''
    orig_size = int(scale * 200.)
    ul = (center - (orig_size / 2.)).astype(int)

    crop_size = int(orig_size * crop_scale_factor)

    if axis == 'all':
        h_start = np.random.rand()
        w_start = np.random.rand()
    elif axis == 'x':
        h_start = np.random.rand()
        w_start = 0.5
    elif axis == 'y':
        h_start = 0.5
        w_start = np.random.rand()
    else:
        raise ValueError(f'axis {axis} is undefined!')

    x1, y1, x2, y2 = get_random_crop_coords(
        height=orig_size,
        width=orig_size,
        crop_height=crop_size,
        crop_width=crop_size,
        h_start=h_start,
        w_start=w_start,
    )
    scale = (y2 - y1) / 200.
    center = ul + np.array([(y1 + y2) / 2, (x1 + x2) / 2])
    return center, scale


def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = resize(
        img, crop_shape
    )  #, interp='nearest') # scipy.misc.imresize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1],
                                                        new_x[0]:new_x[1]]
    return new_img


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)),
                   np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:, 0] = -kp[:, 0]
    return kp


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225],
                                   device=images.device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406],
                                   device=images.device).reshape(1, 3, 1, 1)
    return images


def read_img(img_fn):
    #  return pil_img.fromarray(
    #  cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB))
    #  with open(img_fn, 'rb') as f:
    #  img = pil_img.open(f).convert('RGB')
    #  return img
    if img_fn.endswith('jpeg') or img_fn.endswith('jpg'):
        try:
            with open(img_fn, 'rb') as f:
                img = np.array(jpeg.JPEG(f).decode())
        except jpeg.JPEGRuntimeError:
            # logger.warning('{} produced a JPEGRuntimeError', img_fn)
            img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    else:
        #  elif img_fn.endswith('png') or img_fn.endswith('JPG') or img_fn.endswith(''):
        img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def generate_heatmaps_2d(joints,
                         joints_vis,
                         num_joints=24,
                         heatmap_size=56,
                         image_size=224,
                         sigma=1.75):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints, heatmap_size, heatmap_size),
                      dtype=np.float32)

    tmp_size = sigma * 3

    # denormalize joint into heatmap coordinates
    joints = (joints + 1.) * (image_size / 2.)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size)
        img_y = max(0, ul[1]), min(br[1], heatmap_size)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def generate_part_labels(vertices, faces, cam_t, neural_renderer,
                         body_part_texture, K, R, part_bins):
    batch_size = vertices.shape[0]

    body_parts, depth, mask = neural_renderer(
        vertices,
        faces.expand(batch_size, -1, -1),
        textures=body_part_texture.expand(batch_size, -1, -1, -1, -1, -1),
        K=K.expand(batch_size, -1, -1),
        R=R.expand(batch_size, -1, -1),
        t=cam_t.unsqueeze(1),
    )

    render_rgb = body_parts.clone()

    body_parts = body_parts.permute(0, 2, 3, 1)
    body_parts *= 255.  # multiply it with 255 to make labels distant
    body_parts, _ = body_parts.max(-1)  # reduce to single channel

    body_parts = torch.bucketize(
        body_parts.detach(), part_bins,
        right=True)  # np.digitize(body_parts, bins, right=True)

    # add 1 to make background label 0
    body_parts = body_parts.long() + 1
    body_parts = body_parts * mask.detach()

    return body_parts.long(), render_rgb


def generate_heatmaps_2d_batch(joints,
                               num_joints=24,
                               heatmap_size=56,
                               image_size=224,
                               sigma=1.75):
    batch_size = joints.shape[0]

    joints = joints.detach().cpu().numpy()
    joints_vis = np.ones_like(joints)

    heatmaps = []
    heatmaps_vis = []
    for i in range(batch_size):
        hm, hm_vis = generate_heatmaps_2d(joints[i], joints_vis[i], num_joints,
                                          heatmap_size, image_size, sigma)
        heatmaps.append(hm)
        heatmaps_vis.append(hm_vis)

    return torch.from_numpy(np.stack(heatmaps)).float().to('cuda'), \
           torch.from_numpy(np.stack(heatmaps_vis)).float().to('cuda')


def get_body_part_texture(faces, n_vertices=6890, non_parametric=False):
    smpl_segmentation = joblib.load('data/smpl_partSegmentation_mapping.pkl')

    smpl_vert_idx = smpl_segmentation['smpl_index']
    nparts = 24.

    if non_parametric:
        # reduce the number of body_parts to 14
        # by mapping some joints to others
        nparts = 14.
        joint_mapping = map_smpl_to_common()

        for jm in joint_mapping:
            for j in jm[0]:
                smpl_vert_idx[smpl_vert_idx == j] = jm[1]

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, :3] = smpl_vert_idx[..., None]

    vertex_colors = color.to_rgba(vertex_colors)
    face_colors = vertex_colors[faces].min(axis=1)

    texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
    texture[0, :, 0, 0, 0, :] = face_colors[:, :3] / nparts
    texture = torch.from_numpy(texture).float()
    return texture


def get_default_camera(focal_length, img_size):
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1
    K[0, 2] = img_size / 2.
    K[1, 2] = img_size / 2.
    K = K[None, :, :]
    R = torch.eye(3)[None, :, :]
    return K, R


def read_exif_data(img_fname):
    import PIL.Image
    import PIL.ExifTags

    img = PIL.Image.open(img_fname)
    exif_data = img._getexif()

    if exif_data == None:
        return None

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif_data.items() if k in PIL.ExifTags.TAGS
    }
    return exif
