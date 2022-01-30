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
import time
import shutil
import os.path as osp
from loguru import logger
from shutil import copytree, ignore_patterns


def copy_code(output_folder, curr_folder, code_folder='code'):
    '''
    Copies current state of the code to the log folder and compresses it
    '''
    code_folder = osp.join(output_folder, code_folder)
    if not osp.exists(code_folder):
        os.makedirs(code_folder)

    # Copy code
    logger.info('Copying main files ...')

    for f in [x for x in os.listdir(curr_folder) if x.endswith('.py')]:
        mainpy_path = osp.join(curr_folder, f)
        dest_mainpy_path = osp.join(code_folder, f)
        shutil.copy2(mainpy_path, dest_mainpy_path)

    logger.info('Copying the rest of the source code ...')
    for f in ['pare', 'tests', 'configs', 'scripts', 'cam_reg']:
        src_folder = osp.join(curr_folder, f)
        dest_folder = osp.join(code_folder, osp.split(src_folder)[1])
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        shutil.copytree(src_folder,
                        dest_folder,
                        ignore=ignore_patterns('*.pyc', 'tmp*', '__pycache__'))

    logger.info(
        f'Compressing code folder to {os.path.join(output_folder, "code.zip")}'
    )
    shutil.make_archive(os.path.join(output_folder, 'code'), 'zip',
                        code_folder)

    logger.info(f'Removing {code_folder}')
    shutil.rmtree(code_folder)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print(f'Function execution took {end - start:.3f} seconds.')
        return val

    return wrapper
