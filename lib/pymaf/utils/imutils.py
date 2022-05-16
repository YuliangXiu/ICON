"""
This file contains functions that are used to perform data augmentation.
"""
from turtle import reset
import cv2
import io
import torch
import numpy as np
import scipy.misc
from PIL import Image
from rembg.bg import remove
import human_det

from lib.pymaf.core import constants
from lib.pymaf.utils.streamer import aug_matrix
from torchvision import transforms


def load_img(img_file):
    
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
    return img
    


def get_bbox(img, det):

    input = np.float32(img)
    input = (input / 255.0 -
             (0.5, 0.5, 0.5)) / (0.5, 0.5, 0.5)  # TO [-1.0, 1.0]
    input = input.transpose(2, 0, 1)  # TO [3 x H x W]
    bboxes, probs = det(torch.from_numpy(input).float().unsqueeze(0))

    probs = probs.unsqueeze(3)
    bboxes = (bboxes * probs).sum(dim=1, keepdim=True) / probs.sum(
        dim=1, keepdim=True)
    bbox = bboxes[0, 0, 0].cpu().numpy()

    return bbox


def get_transformer(input_res):
    
    image_to_tensor = transforms.Compose([
        transforms.Resize(input_res),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_to_tensor = transforms.Compose([
        transforms.Resize(input_res),
        transforms.ToTensor(),
        transforms.Normalize((0.0, ), (1.0, ))
    ])

    image_to_pymaf_tensor = transforms.Compose([
        transforms.Resize(size=224),
        transforms.Normalize(mean=constants.IMG_NORM_MEAN,
                             std=constants.IMG_NORM_STD)
    ])
    
    image_to_pixie_tensor = transforms.Compose([
        transforms.Resize(224)
    ])
    def image_to_hybrik_tensor(img):
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)
        return img

    return [image_to_tensor, mask_to_tensor, image_to_pymaf_tensor, image_to_pixie_tensor, image_to_hybrik_tensor]


def process_image(img_file, det, hps_type, input_res=512, device=None):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    
    [image_to_tensor, mask_to_tensor, image_to_pymaf_tensor, image_to_pixie_tensor, image_to_hybrik_tensor] = get_transformer(input_res)

    img_ori = load_img(img_file)
    
    in_height, in_width, _ = img_ori.shape
    M = aug_matrix(in_width, in_height, input_res*2, input_res*2)
    
    # from rectangle to square
    img_for_crop = cv2.warpAffine(img_ori, M[0:2, :], 
                                    (input_res*2, input_res*2), flags=cv2.INTER_CUBIC)
    
    if det is not None:

        # detection for bbox
        bbox = get_bbox(img_for_crop, det)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center = np.array([(bbox[0] + bbox[2]) / 2.0,
                           (bbox[1] + bbox[3]) / 2.0])

    else:
        
        # Assume that the person is centerered in the image
        height = img_for_crop.shape[0]
        width = img_for_crop.shape[1]
        center = np.array([width // 2, height // 2])

    scale = max(height, width) / 180
    if hps_type == 'hybrik':
        img_np = crop_for_hybrik(img_for_crop, center, np.array([scale * 180, scale * 180]))
    else:
        img_np = crop(img_for_crop, center, scale, (input_res, input_res))

    with torch.no_grad():
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format='png')
        img_pil = Image.open(
            io.BytesIO(remove(buf.getvalue()))).convert("RGBA")

    # for icon
    img_rgb = image_to_tensor(img_pil.convert("RGB"))
    img_mask = torch.tensor(1.0) - (mask_to_tensor(img_pil.split()[-1]) <
                                    torch.tensor(0.5)).float()
    img_tensor = img_rgb * img_mask

    # for hps
    img_hps = img_np.astype(np.float32) / 255.
    img_hps = torch.from_numpy(img_hps).permute(2, 0, 1)
    
    if hps_type == 'bev':
        img_hps = img_np[:,:,[2,1,0]]
    elif hps_type == 'hybrik':
        img_hps = image_to_hybrik_tensor(img_hps).unsqueeze(0).to(device)
    elif hps_type != 'pixie':
        img_hps = image_to_pymaf_tensor(img_hps).unsqueeze(0).to(device)
    else:
        img_hps = image_to_pixie_tensor(img_hps).unsqueeze(0).to(device)
    
    # uncrop params
    uncrop_param = {'center': center,
                    'scale': scale,
                    'ori_shape': img_ori.shape,
                    'box_shape': img_np.shape,
                    'crop_shape': img_for_crop.shape,
                    'M': M}
    
    return img_tensor, img_hps, img_ori, img_mask, uncrop_param

def get_transform(center, scale, res):
    
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    
    return t


def transform(pt, center, scale, res, invert=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.around(new_pt[:2]).astype(np.int16)


def crop(img, center, scale, res):
    """Crop image according to the supplied bounding box."""
    
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))
    
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
    
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res))

    return new_img

def crop_for_hybrik(img, center, scale):
    inp_h, inp_w = (256, 256)
    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    new_img = cv2.warpAffine(img, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    return new_img

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):

    def get_dir(src_point, rot_rad):
        """Rotate the point by `rot_rad` degree."""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(a, b):
        """Return vector c that perpendicular to (a - b)."""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def corner_align(ul, br):
    
    if ul[1]-ul[0] != br[1]-br[0]:
        ul[1] = ul[0]+br[1]-br[0]
    
    return ul, br

def uncrop(img, center, scale, orig_shape):
    
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    
    res = img.shape[:2]
    
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))
    
    # quick fix
    ul, br = corner_align(ul, br)
    
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    
    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(crop_shape))
    
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    
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


def flip_kp(kp, is_smpl=False):
    """Flip keypoints."""
    if len(kp) == 24:
        if is_smpl:
            flipped_parts = constants.SMPL_JOINTS_FLIP_PERM
        else:
            flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        if is_smpl:
            flipped_parts = constants.SMPL_J49_FLIP_PERM
        else:
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


def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0) / (2 * ratio)

    return kp_2d


def generate_heatmap(joints, heatmap_size, sigma=1, joints_vis=None):
    '''
    param joints:  [num_joints, 3]
    param joints_vis: [num_joints, 3]
    return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = joints.shape[0]
    device = joints.device
    cur_device = torch.device(device.type, device.index)
    if not hasattr(heatmap_size, '__len__'):
        # width  height
        heatmap_size = [heatmap_size, heatmap_size]
    assert len(heatmap_size) == 2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    if joints_vis is not None:
        target_weight[:, 0] = joints_vis[:, 0]
    target = torch.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                         dtype=torch.float32,
                         device=cur_device)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] * heatmap_size[0] + 0.5)
        mu_y = int(joints[joint_id][1] * heatmap_size[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        # x = np.arange(0, size, 1, np.float32)
        # y = x[:, np.newaxis]
        # x0 = y0 = size // 2
        # # The gaussian is not normalized, we want the center value to equal 1
        # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # g = torch.from_numpy(g.astype(np.float32))

        x = torch.arange(0, size, dtype=torch.float32, device=cur_device)
        y = x.unsqueeze(-1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight
