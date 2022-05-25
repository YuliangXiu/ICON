'''
crop
for numpy array
Given image, bbox(center, bboxsize)
return: cropped image, tform(used for transform the keypoint accordingly)

only support crop to squared images
'''

import numpy as np
from skimage.transform import estimate_transform, warp, resize, rescale


def points2bbox(points, points_scale=None):
    # recover range
    if points_scale:
        points[:, 0] = points[:, 0]*points_scale[1]/2 + points_scale[1]/2
        points[:, 1] = points[:, 1]*points_scale[0]/2 + points_scale[0]/2

    left = np.min(points[:, 0])
    right = np.max(points[:, 0])
    top = np.min(points[:, 1])
    bottom = np.max(points[:, 1])
    size = max(right - left, bottom - top)
    # + old_size*0.1])
    center = np.array([right - (right - left) / 2.0,
                      bottom - (bottom - top) / 2.0])
    return center, size
    # translate center


def augment_bbox(center, bbox_size, scale=[1.0, 1.0], trans_scale=0.):
    trans_scale = (np.random.rand(2)*2 - 1) * trans_scale
    center = center + trans_scale*bbox_size  # 0.5
    scale = np.random.rand() * (scale[1] - scale[0]) + scale[0]
    size = int(bbox_size*scale)
    return center, size


def crop_array(image, center, bboxsize, crop_size):
    ''' for single image only
    Args:
        image (numpy.Array): the reference array of shape HxWXC.
        size (Tuple[int, int]): a tuple with the height and width that will be
          used to resize the extracted patches.
    Returns:
        cropped_image
        tform: 3x3 affine matrix
    '''
    # points: top-left, top-right, bottom-right
    src_pts = np.array([
        [center[0]-bboxsize/2, center[1]-bboxsize/2],
        [center[0] + bboxsize/2, center[1]-bboxsize/2],
        [center[0]+bboxsize/2, center[1]+bboxsize/2]])
    DST_PTS = np.array([[0, 0], [crop_size - 1, 0],
                       [crop_size - 1, crop_size - 1]])

    # estimate transformation between points
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    # warp images
    cropped_image = warp(image, tform.inverse,
                         output_shape=(crop_size, crop_size))

    return cropped_image, tform.params.T


class Cropper(object):
    def __init__(self, crop_size, scale=[1, 1], trans_scale=0.):
        self.crop_size = crop_size
        self.scale = scale
        self.trans_scale = trans_scale

    def crop(self, image, points, points_scale=None):
        # points to bbox
        center, bbox_size = points2bbox(points, points_scale)
        # argument bbox.
        center, bbox_size = augment_bbox(
            center, bbox_size, scale=self.scale, trans_scale=self.trans_scale)
        # crop
        cropped_image, tform = crop_array(
            image, center, bbox_size, self.crop_size)
        return cropped_image, tform
