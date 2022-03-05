# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at pixie@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

class FLAMETex(nn.Module):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """
    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)/255.
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)/255.
        else:
            print('texture type ', config.tex_type, 'not exist!')
            raise NotImplementedError

        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None,...]
        texture_basis = torch.from_numpy(texture_basis[:,:n_tex]).float()[None,...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode=None):
        '''
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        '''
        texture = self.texture_mean + (self.texture_basis*texcode[:,None,:]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0,3,1,2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:,[2,1,0], :,:]
        return texture


def texture_flame2smplx(cached_data, flame_texture, smplx_texture):
    ''' Convert flame texture map (face-only) into smplx texture map (includes body texture)
    TODO: pytorch version ==> grid sample
    '''
    if smplx_texture.shape[0] != smplx_texture.shape[1]:
        print('SMPL-X texture not squared (%d != %d)' % (smplx_texture[0], smplx_texture[1]))
        return
    if smplx_texture.shape[0] != cached_data['target_resolution']:
        print('SMPL-X texture size does not match cached image resolution (%d != %d)' % (smplx_texture.shape[0], cached_data['target_resolution']))
        return
    x_coords = cached_data['x_coords']
    y_coords = cached_data['y_coords']
    target_pixel_ids = cached_data['target_pixel_ids']
    source_uv_points = cached_data['source_uv_points']

    source_tex_coords = np.zeros_like((source_uv_points)).astype(int)
    source_tex_coords[:, 0] = np.clip(flame_texture.shape[0]*(1.0-source_uv_points[:,1]), 0.0, flame_texture.shape[0]).astype(int)
    source_tex_coords[:, 1] = np.clip(flame_texture.shape[1]*(source_uv_points[:,0]), 0.0, flame_texture.shape[1]).astype(int)

    smplx_texture[y_coords[target_pixel_ids].astype(int), x_coords[target_pixel_ids].astype(int), :] = flame_texture[source_tex_coords[:,0], source_tex_coords[:,1]]

    return smplx_texture