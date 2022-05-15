
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

from pdb import set_trace
import numpy as np
import cv2
import torch
import torchvision
import trimesh
from pytorch3d.io import load_obj
import os, sys
from termcolor import colored
import os.path as osp
from scipy.spatial import cKDTree
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from PIL import Image, ImageFont, ImageDraw

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pytorch3d.renderer.mesh import rasterize_meshes
from lib.common.render_utils import Pytorch3dRasterizer, face_vertices
from lib.pymaf.utils.imutils import uncrop
from pytorch3d.structures import Meshes



def get_mask(tensor, dim):

    mask = torch.abs(tensor).sum(dim=dim, keepdims=True) > 0.0
    mask = mask.type_as(tensor)

    return mask

def blend_rgb_norm(rgb, norm, mask):
    
    # [0,0,0] or [127,127,127] should be marked as mask
    final = rgb * (1-mask) + norm * (mask)
    
    return final.astype(np.uint8)

def unwrap(image, data):
    
    img_uncrop = uncrop(np.array(Image.fromarray(image).resize(data['uncrop_param']['box_shape'][:2])), 
                                 data['uncrop_param']['center'], 
                                 data['uncrop_param']['scale'], 
                                 data['uncrop_param']['crop_shape'])
            
    
    img_orig = cv2.warpAffine(img_uncrop,
                        np.linalg.inv(data['uncrop_param']['M'])[:2, :],
                        data['uncrop_param']['ori_shape'][::-1][1:],
                        flags=cv2.INTER_CUBIC)
    
    return img_orig


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):

    # and (b) the edge length of the predicted mesh
    losses["edge"]['value'] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["normal"]['value'] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["laplacian"]['value'] = mesh_laplacian_smoothing(mesh,
                                                            method="uniform")


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def load_checkpoint(model, cfg):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}

    device = torch.device(f"cuda:{cfg['test_gpus'][0]}")

    if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
        main_dict = torch.load(cfg.resume_path,
                               map_location=device)['state_dict']

        main_dict = {
            k: v
            for k, v in main_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and (
                'reconEngine' not in k) and ("normal_filter" not in k) and (
                    'voxelization' not in k)
        }
        print(colored(f"Resume MLP weights from {cfg.resume_path}", 'green'))

    if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
        normal_dict = torch.load(cfg.normal_path,
                                 map_location=device)['state_dict']

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key,
                                 key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {cfg.normal_path}", 'green'))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    model.netG = model.netG.to(device)
    model.reconEngine = model.reconEngine.to(device)

    model.netG.training = False
    model.netG.eval()

    del main_dict
    del normal_dict
    del model_dict

    return model


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, 'vertices.txt'))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, 'faces.txt'),
                            dtype=np.int32) - 1
    smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] +
                      smpl_vertex_code[smpl_faces[:, 1]] +
                      smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
    smpl_tetras = np.loadtxt(os.path.join(folder, 'tetrahedrons.txt'),
                             dtype=np.int32) - 1

    return smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras


def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = torch.tile((1-select),(1,dim,1))*dim + \
        torch.arange(0,dim).unsqueeze(0).unsqueeze(2).type_as(select)
    feat_select = torch.gather(feat, 1, idx.long())

    return feat_select


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


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """ 
    #(p, q, u, v)
    v0, v1, v2 = vertices[:,0], vertices[:,0], vertices[:,0]
    p = points
    
    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def cal_sdf_batch(verts, faces, cmaps, vis, points):
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    Bsize = points.shape[0]
    
    normals = Meshes(verts, faces).verts_normals_padded()
    
    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)
    cmaps = face_vertices(cmaps, faces)
    vis = face_vertices(vis, faces)
    
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    closest_triangles = torch.gather(triangles, 1, pts_ind[:,:,None,None].expand(-1,-1,3,3)).view(-1,3,3)
    closest_normals = torch.gather(normals, 1, pts_ind[:,:,None,None].expand(-1,-1,3,3)).view(-1,3,3)
    closest_cmaps = torch.gather(cmaps, 1, pts_ind[:,:,None,None].expand(-1,-1,3,3)).view(-1,3,3)
    closest_vis = torch.gather(vis, 1, pts_ind[:,:,None,None].expand(-1,-1,3,1)).view(-1,3,1)
    bary_weights = barycentric_coordinates_of_projection(points.view(-1,3), closest_triangles)
    
    pts_cmap = (closest_cmaps*bary_weights[:,:,None]).sum(1).unsqueeze(0)
    pts_vis = (closest_vis*bary_weights[:,:,None]).sum(1).unsqueeze(0).ge(1e-1)
    pts_norm = (closest_normals*bary_weights[:,:,None]).sum(1).unsqueeze(0) * torch.tensor([-1.0, 1.0, -1.0]).type_as(normals)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1), pts_norm.view(Bsize, -1, 3), pts_cmap.view(Bsize, -1, 3), pts_vis.view(Bsize, -1, 1)


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def projection(points, calib, format='numpy'):
    if format == 'tensor':
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def load_obj_mesh_for_Hoppe(mesh_file):
    vertex_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    normals, _ = compute_normal(vertices, faces)

    return vertices, normals, faces


def load_obj_mesh_with_color(mesh_file):
    vertex_data = []
    color_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            c = list(map(float, values[4:7]))
            color_data.append(c)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    colors = np.array(color_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    return vertices, colors, faces


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[1]),
                            [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[2]),
                            [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data)
        face_uvs[face_uvs > 0] -= 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms, _ = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data)
            face_normals[face_normals > 0] -= 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    vert_norms = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    face_norms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(face_norms)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vert_norms[faces[:, 0]] += face_norms
    vert_norms[faces[:, 1]] += face_norms
    vert_norms[faces[:, 2]] += face_norms
    normalize_v3(vert_norms)

    return vert_norms, face_norms


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                   (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def calculate_mIoU(outputs, labels):

    SMOOTH = 1e-6

    outputs = outputs.int()
    labels = labels.int()

    intersection = (
        outputs
        & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(
        20 * (iou - 0.5), 0,
        10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean().detach().cpu().numpy(
    )  # Or thresholded.mean() if you are interested in average across the batch


def mask_filter(mask, number=1000):
    """only keep {number} True items within a mask

    Args:
        mask (bool array): [N, ]
        number (int, optional): total True item. Defaults to 1000.
    """
    true_ids = np.where(mask)[0]
    keep_ids = np.random.choice(true_ids, size=number)
    filter_mask = np.isin(np.arange(len(mask)), keep_ids)

    return filter_mask


def query_mesh(path):

    verts, faces_idx, _ = load_obj(path)

    return verts, faces_idx.verts_idx


def add_alpha(colors, alpha=0.7):

    colors_pad = np.pad(colors, ((0, 0), (0, 1)),
                        mode='constant',
                        constant_values=alpha)

    return colors_pad


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type='smpl'):

    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0),
                                           nrow=nrow)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 *
         255.0).astype(np.uint8))
    
    # add text
    draw = ImageDraw.Draw(grid_img)
    grid_size = 512
    if loss is not None:
        draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)
        
    if type == 'smpl':
        for col_id, col_txt in enumerate(
            ['image', 'smpl-norm(render)', 'cloth-norm(pred)', 'diff-norm', 'diff-mask']):
            draw.text((10+(col_id*grid_size), 5), col_txt, (255, 0, 0), font=font)
    elif type == 'cloth':
        for col_id, col_txt in enumerate(
            ['image', 'cloth-norm(recon)', 'cloth-norm(pred)', 'diff-norm']):
            draw.text((10+(col_id*grid_size), 5), col_txt, (255, 0, 0), font=font)
        for col_id, col_txt in enumerate(
            ['0', '90', '180', '270']):
            draw.text((10+(col_id*grid_size), grid_size*2+5), col_txt, (255, 0, 0), font=font)
    else:
        print(f"{type} should be 'smpl' or 'cloth'")
        

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]),
                               Image.ANTIALIAS)

    return grid_img


def clean_mesh(verts, faces):

    device = verts.device

    mesh_lst = trimesh.Trimesh(verts.detach().cpu().numpy(),
                               faces.detach().cpu().numpy())
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

    final_verts = torch.as_tensor(mesh_clean.vertices).float().to(device)
    final_faces = torch.as_tensor(mesh_clean.faces).int().to(device)

    return final_verts, final_faces


def merge_mesh(verts_A, faces_A, verts_B, faces_B):

    sep_mesh = trimesh.Trimesh(np.concatenate([verts_A, verts_B], axis=0),
                               np.concatenate(
                                   [faces_A, faces_B + faces_A.max() + 1],
                                   axis=0),
                               maintain_order=True,
                               process=False)

    colors = np.ones_like(sep_mesh.vertices)
    colors[:verts_A.shape[0]] *= np.array([255.0, 0.0, 0.0])
    colors[verts_A.shape[0]:] *= np.array([0.0, 255.0, 0.0])
    sep_mesh.visual.vertex_colors = colors

    # union_mesh = trimesh.boolean.union([trimesh.Trimesh(verts_A, faces_A),
    #                                     trimesh.Trimesh(verts_B, faces_B)], engine='blender')

    return sep_mesh


def mesh_move(mesh_lst, step, scale=1.0):

    trans = np.array([1.0, 0.0, 0.0]) * step

    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=trans)

    results = []

    for mesh in mesh_lst:
        mesh.apply_transform(resize_matrix)
        results.append(mesh)

    return results


class SMPLX():
    def __init__(self):

        self.current_dir = osp.join(osp.dirname(__file__),
                                    "../../data/smpl_related")

        self.smpl_verts_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_verts.npy")
        self.smplx_verts_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_verts.npy")
        self.faces_path = osp.join(self.current_dir,
                                   "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir,
                                       "smpl_data/smplx_cmap.npy")

        self.faces = np.load(self.faces_path)
        self.verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

    def get_smpl_mat(self, vert_ids):

        mat = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        return mat[vert_ids, :]

    def smpl2smplx(self, vert_ids=None):
        """convert vert_ids in smpl to vert_ids in smplx

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smplx_tree = cKDTree(self.verts, leafsize=1)
        _, ind = smplx_tree.query(self.smpl_verts, k=1)  # ind: [smpl_num, 1]

        if vert_ids is not None:
            smplx_vert_ids = ind[vert_ids]
        else:
            smplx_vert_ids = ind

        return smplx_vert_ids

    def smplx2smpl(self, vert_ids=None):
        """convert vert_ids in smplx to vert_ids in smpl

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smpl_tree = cKDTree(self.smpl_verts, leafsize=1)
        _, ind = smpl_tree.query(self.verts, k=1)  # ind: [smplx_num, 1]
        if vert_ids is not None:
            smpl_vert_ids = ind[vert_ids]
        else:
            smpl_vert_ids = ind

        return smpl_vert_ids
