
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

import numpy as np
import smplx
import trimesh
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from lib.common.render_utils import face_vertices
from lib.dataset.mesh_util import SMPLX

get_smpl_model = lambda smpl_type, smpl_gender : smplx.create(
            model_path=SMPLX().model_dir,
            gender=smpl_gender,
            model_type=smpl_type,
            use_pca=True,
            num_pca_comps=12,
            ext='npz')


def normalization(data):
    _range = np.max(data) - np.min(data)
    return ((data - np.min(data)) / _range)


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0), return_vf=False):

    fitted_body = trimesh.load(fitted_path, process=False, maintain_order=True)
    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    if return_vf:
        return np.array(fitted_body.vertices), np.array(fitted_body.faces)
    else:
        return fitted_body
    
def load_fit_body(fitted_path, scale, smpl_type='smplx', smpl_gender='neutral'):
    
    param = np.load(fitted_path, allow_pickle=True)
    for key in param.keys():
        param[key] = torch.as_tensor(param[key])
        
    smpl_model = get_smpl_model(smpl_type, smpl_gender)
    smpl_out = smpl_model(shape_params=param['betas'],
                            expression_params=param['expression'],
                            body_pose=param['body_pose'],
                            global_pose=param['global_orient'],
                            jaw_pose=param['jaw_pose'],
                            left_hand_pose=param['left_hand_pose'],
                            right_hand_pose=param['right_hand_pose'],
                            leye_pose=param['leye_pose'],
                            reye_pose=param['reye_pose'])
    
    print(f"smpl-x scale: {param['scale']}")
    smpl_verts = ((smpl_out.vertices[0] * param['scale'] + param['translation']) * scale).detach()
    smpl_joints = ((smpl_out.joints[0] * param['scale'] + param['translation']) * scale).detach()
    smpl_mesh = trimesh.Trimesh(smpl_verts, 
                                smpl_model.faces, 
                                process=False, maintain_order=True)
    
    return smpl_mesh, smpl_joints
    


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


# https://github.com/ratcave/wavefront_reader
def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(' ')

                    # assume texture maps are in the same level
                    # WARNING: do not include space in your filename!!
                    if 'map' in prefix:
                        material[prefix] = split_data[-1].split('\\')[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials


def load_obj_mesh_mtl(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    # face per material
    face_data_mat = {}
    face_norm_data_mat = {}
    face_uv_data_mat = {}

    # current material name
    mtl_data = None
    cur_mat = None

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
        elif values[0] == 'mtllib':
            mtl_data = read_mtlfile(
                mesh_file.replace(mesh_file.split('/')[-1], values[1]))
        elif values[0] == 'usemtl':
            cur_mat = values[1]
        elif values[0] == 'f':
            # local triangle data
            l_face_data = []
            l_face_uv_data = []
            l_face_norm_data = []

            # quad mesh
            if len(values) > 4:
                f = list(
                    map(
                        lambda x: int(x.split('/')[0]) if int(x.split('/')[0])
                        < 0 else int(x.split('/')[0]) - 1, values[1:4]))
                l_face_data.append(f)
                f = list(
                    map(
                        lambda x: int(x.split('/')[0])
                        if int(x.split('/')[0]) < 0 else int(x.split('/')[0]) -
                        1, [values[3], values[4], values[1]]))
                l_face_data.append(f)
            # tri mesh
            else:
                f = list(
                    map(
                        lambda x: int(x.split('/')[0]) if int(x.split('/')[0])
                        < 0 else int(x.split('/')[0]) - 1, values[1:4]))
                l_face_data.append(f)
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(
                        map(
                            lambda x: int(x.split('/')[1])
                            if int(x.split('/')[1]) < 0 else int(
                                x.split('/')[1]) - 1, values[1:4]))
                    l_face_uv_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split('/')[1])
                            if int(x.split('/')[1]) < 0 else int(
                                x.split('/')[1]) - 1,
                            [values[3], values[4], values[1]]))
                    l_face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(
                        map(
                            lambda x: int(x.split('/')[1])
                            if int(x.split('/')[1]) < 0 else int(
                                x.split('/')[1]) - 1, values[1:4]))
                    l_face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(
                        map(
                            lambda x: int(x.split('/')[2])
                            if int(x.split('/')[2]) < 0 else int(
                                x.split('/')[2]) - 1, values[1:4]))
                    l_face_norm_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split('/')[2])
                            if int(x.split('/')[2]) < 0 else int(
                                x.split('/')[2]) - 1,
                            [values[3], values[4], values[1]]))
                    l_face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(
                        map(
                            lambda x: int(x.split('/')[2])
                            if int(x.split('/')[2]) < 0 else int(
                                x.split('/')[2]) - 1, values[1:4]))
                    l_face_norm_data.append(f)

            face_data += l_face_data
            face_uv_data += l_face_uv_data
            face_norm_data += l_face_norm_data

            if cur_mat is not None:
                if cur_mat not in face_data_mat.keys():
                    face_data_mat[cur_mat] = []
                if cur_mat not in face_uv_data_mat.keys():
                    face_uv_data_mat[cur_mat] = []
                if cur_mat not in face_norm_data_mat.keys():
                    face_norm_data_mat[cur_mat] = []
                face_data_mat[cur_mat] += l_face_data
                face_uv_data_mat[cur_mat] += l_face_uv_data
                face_norm_data_mat[cur_mat] += l_face_norm_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    norms = np.array(norm_data)
    norms = normalize_v3(norms)
    face_normals = np.array(face_norm_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, norms, face_normals, uvs, face_uvs)

    if cur_mat is not None and mtl_data is not None:
        for key in face_data_mat:
            face_data_mat[key] = np.array(face_data_mat[key])
            face_uv_data_mat[key] = np.array(face_uv_data_mat[key])
            face_norm_data_mat[key] = np.array(face_norm_data_mat[key])

        out_tuple += (face_data_mat, face_norm_data_mat, face_uv_data_mat,
                      mtl_data)

    return out_tuple


def load_scan(mesh_file, with_normal=False, with_texture=False):
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
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
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
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def compute_normal_batch(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    vert_norm = torch.zeros(bs * nv, 3).type_as(vertices)
    tris = face_vertices(vertices, faces)
    face_norm = F.normalize(torch.cross(tris[:, :, 1] - tris[:, :, 0],
                                        tris[:, :, 2] - tris[:, :, 0]),
                            dim=-1)

    faces = (faces +
             (torch.arange(bs).type_as(faces) * nv)[:, None, None]).view(
                 -1, 3)

    vert_norm[faces[:, 0]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 1]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 2]] += face_norm.view(-1, 3)

    vert_norm = F.normalize(vert_norm, dim=-1).view(bs, nv, 3)

    return vert_norm


# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs, faceuvs):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    # NOTE: traditional version is below

    # pts_tris = vertices[faces]
    # uv_tris = uvs[faceuvs]

    # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)

    # for i in range(W.shape[0]):
    #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tan[faces[:,0]] += W[:,:,0]
    # tan[faces[:,1]] += W[:,:,0]
    # tan[faces[:,2]] += W[:,:,0]

    # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # btan[faces[:,0]] += W[:,:,1]
    # btan[faces[:,1]] += W[:,:,1]
    # btan[faces[:,2]] += W[:,:,1]

    # normalize_v3(tan)

    # ndott = np.sum(normals*tan, 1, keepdims=True)
    # tan = tan - ndott * normals

    # normalize_v3(btan)
    # normalize_v3(tan)

    # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    return tan, btan
