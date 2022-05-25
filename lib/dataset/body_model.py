
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
import pickle
import torch
import os


class SMPLModel():
    def __init__(self, model_path, age):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')

            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])
            self.v_template = np.asarray(params['v_template'])
            self.shapedirs = np.asarray(params['shapedirs'])
            self.faces = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        if age == 'kid':
            v_template_smil = np.load(
                os.path.join(os.path.dirname(model_path),
                             "smpl/smpl_kid_template.npy"))
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - self.v_template,
                                             axis=2)
            self.shapedirs = np.concatenate(
                (self.shapedirs[:, :, :self.beta_shape[0]], v_template_diff),
                axis=2)
            self.beta_shape[0] += 1

        id_to_col = {
            self.kintree_table[1, i]: i
            for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None
        self.G = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0),
                                 (self.R.shape[0] - 1, 3, 3))
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(
            np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack([
                        self.R[i],
                        ((self.J[i, :] - self.J[self.parent[i], :]).reshape(
                            [3, 1]))
                    ])))
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])))
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1,
                                                                    4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        self.G = G

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick
        ]).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0),
                                 [theta.shape[0], 3, 3])
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class TetraSMPLModel():
    def __init__(self,
                 model_path,
                 model_addition_path,
                 age='adult',
                 v_template=None):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')

            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])

            if v_template is not None:
                self.v_template = v_template
            else:
                self.v_template = np.asarray(params['v_template'])

            self.shapedirs = np.asarray(params['shapedirs'])
            self.faces = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])

        params_added = np.load(model_addition_path)
        self.v_template_added = params_added['v_template_added']
        self.weights_added = params_added['weights_added']
        self.shapedirs_added = params_added['shapedirs_added']
        self.posedirs_added = params_added['posedirs_added']
        self.tetrahedrons = params_added['tetrahedrons']

        id_to_col = {
            self.kintree_table[1, i]: i
            for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        if age == 'kid':
            v_template_smil = np.load(
                os.path.join(os.path.dirname(model_path),
                             "smpl/smpl_kid_template.npy"))
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - self.v_template,
                                             axis=2)
            self.shapedirs = np.concatenate(
                (self.shapedirs[:, :, :self.beta_shape[0]], v_template_diff),
                axis=2)
            self.beta_shape[0] += 1

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.verts_added = None
        self.J = None
        self.R = None
        self.G = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """

        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        if torch.is_tensor(beta):
            beta = beta.detach().cpu().numpy()

        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        v_shaped_added = self.shapedirs_added.dot(
            self.beta) + self.v_template_added
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0),
                                 (self.R.shape[0] - 1, 3, 3))
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        v_posed_added = v_shaped_added + self.posedirs_added.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(
            np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack([
                        self.R[i],
                        ((self.J[i, :] - self.J[self.parent[i], :]).reshape(
                            [3, 1]))
                    ])))
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])))
        self.G = G
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1,
                                                                    4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        T_added = np.tensordot(self.weights_added, G, axes=[[1], [0]])
        rest_shape_added_h = np.hstack(
            (v_posed_added, np.ones([v_posed_added.shape[0], 1])))
        v_added = np.matmul(T_added,
                            rest_shape_added_h.reshape([-1, 4,
                                                        1])).reshape([-1, 4
                                                                      ])[:, :3]
        self.verts_added = v_added + self.trans.reshape([1, 3])

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick
        ]).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0),
                                 [theta.shape[0], 3, 3])
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_mesh_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def save_tetrahedron_to_obj(self, path):
        """
        Save the tetrahedron SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """

        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f 1 0 0\n' % (v[0], v[1], v[2]))
            for va in self.verts_added:
                fp.write('v %f %f %f 0 0 1\n' % (va[0], va[1], va[2]))
            for t in self.tetrahedrons + 1:
                fp.write('f %d %d %d\n' % (t[0], t[2], t[1]))
                fp.write('f %d %d %d\n' % (t[0], t[3], t[2]))
                fp.write('f %d %d %d\n' % (t[0], t[1], t[3]))
                fp.write('f %d %d %d\n' % (t[1], t[2], t[3]))
