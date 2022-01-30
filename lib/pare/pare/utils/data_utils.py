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

import os
import numpy as np
from skimage.util.shape import view_as_windows


def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen, ), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices


def combine_npz_files(dir, out_file, add_indices=True):
    npz_files = sorted(
        [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.npz')])

    d = {}
    dout = {}

    # Loop over the source files
    for i, s in enumerate(npz_files):
        print(f'hadd Source file {i}: {s}')
        with np.load(s) as data:
            if i == 0:
                for k in data.files:
                    d[k] = []
            # Loop over all the keys
            for k in data.files:
                d[k].append(data[k])

    # Merge arrays via np.vstack()
    print('hadding...')
    for k, v in d.items():
        vv = np.concatenate(v)
        dout[k] = vv

    if add_indices:
        print('Adding indices to the dataset...')
        print('num samples', dout['imgname'].shape[0])
        dout['indices'] = np.arange(dout['imgname'].shape[0])

    # Write to the target file
    np.savez(out_file, **dout)
    print(f'Saved file {out_file}')
