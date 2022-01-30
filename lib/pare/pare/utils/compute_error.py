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

import sys
import joblib
import numpy as np
from loguru import logger
from pare.utils.kp_utils import get_common_joint_names
from pare.core.constants import pw3d_occluded_sequences, pw3d_test_sequences

joint_names = get_common_joint_names()
occluded_sequences = pw3d_occluded_sequences
test_sequences = pw3d_test_sequences


def compute_error(result_file):
    results = joblib.load(result_file)

    logger.info(f'MPJPE on 3DPW-All Set: {results["mpjpe"].mean() * 1000}')
    logger.info(
        f'PA-MPJPE on 3DPW-All Set: {results["pampjpe"].mean() * 1000}')
    logger.info(f'PVE on 3DPW-All Set: {results["v2v"].mean() * 1000}')

    ######## TEST SET #########

    seq_names = [x.split('/')[-2] for x in results['imgname']]

    mpjpe = []
    pampjpe = []
    v2v = []

    ss = []
    for idx, seq in enumerate(seq_names):

        if seq in test_sequences:
            mpjpe.append(results['mpjpe'][idx].mean())
            pampjpe.append(results['pampjpe'][idx].mean())
            v2v.append(results['v2v'][idx])
            ss.append(seq)

    mpjpe = np.array(mpjpe)
    pampjpe = np.array(pampjpe)
    v2v = np.array(v2v)

    logger.info(f'MPJPE on 3DPW-Test Set: {np.array(mpjpe).mean() * 1000}')
    logger.info(
        f'PA-MPJPE on 3DPW-Test Set: {np.array(pampjpe).mean() * 1000}')
    logger.info(f'PVE on 3DPW-Test Set: {np.array(v2v).mean() * 1000}')

    ######## OCCLUSION #########
    # occluded error
    seq_names = [x.split('/')[-2] for x in results['imgname']]

    mpjpe = []
    pampjpe = []
    v2v = []

    ss = []
    for idx, seq in enumerate(seq_names):
        seq = '_'.join(seq.split('_')[:2])

        if seq in occluded_sequences:
            mpjpe.append(results['mpjpe'][idx].mean())
            pampjpe.append(results['pampjpe'][idx].mean())
            v2v.append(results['v2v'][idx])
            ss.append(seq)

    mpjpe = np.array(mpjpe)
    pampjpe = np.array(pampjpe)
    v2v = np.array(v2v)

    logger.info(
        f'MPJPE on Occluded Sequences: {np.array(mpjpe).mean() * 1000}')
    logger.info(
        f'PA-MPJPE on Occluded Sequences: {np.array(pampjpe).mean() * 1000}')
    logger.info(f'PVE on Occluded Sequences: {np.array(v2v).mean() * 1000}')
