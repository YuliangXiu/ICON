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
import torch
import subprocess
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from collections import OrderedDict

from ..core.config import update_hparams, SMPL_MEAN_PARAMS


def auto_lr_finder(model, trainer):
    logger.info('Running auto learning rate finder')

    # Run learning rate finder
    lr_finder = trainer.lr_find(model)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    logger.info(f'Found new learning rate {new_lr}')

    # update hparams of the model
    model.hparams.lr = new_lr


def load_pretrained_model(model,
                          state_dict,
                          strict=False,
                          overwrite_shape_mismatch=True,
                          remove_lightning=False):
    if remove_lightning:
        logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(
                            f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                            f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}'
                        )

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:, -7:]],
                                dim=-1)
                            logger.warning(
                                f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} '
                            )
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError(
                'there are shape inconsistencies between pretrained ckpt and current ckpt'
            )
    return model


def resume_training(args, script='train.py'):
    cmd = [
        'python',
        script,
        '--cfg',
        args.cfg,
    ]
    if args.cluster:
        cmd.append('--cluster')

    cmd += ['--gpu_min_mem', str(args.gpu_min_mem)]
    cmd += ['--memory', str(args.memory)]

    opts = args.opts

    hparams = update_hparams(args.cfg)

    if not 'TRAINING.RESUME' in args.opts:
        ckpt_files = []
        for root, dirs, files in os.walk(hparams.LOG_DIR, topdown=False):
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_files.append(os.path.join(root, f))

        epoch_idx = [int(x.split('=')[-1].split('.')[0]) for x in ckpt_files]
        last_epoch_idx = np.argsort(epoch_idx)[-1]
        ckpt_file = ckpt_files[last_epoch_idx]

        opts += ['LOG_DIR', 'logs/']
        if args.resume_wo_optimizer:
            opts += ['TRAINING.PRETRAINED_LIT', ckpt_file]
        else:
            opts += ['TRAINING.RESUME', ckpt_file]
        # opts += ['TRAINING.PRETRAINED_LIT', 'null']
        # opts += ['TRAINING.PRETRAINED', 'null']

    cmd += ['--opts'] + opts

    logger.info(f'Running cmd: \"{" ".join(cmd)}\"')

    subprocess.call(cmd)
    exit(0)


def parse_datasets_ratios(datasets_and_ratios):
    s_ = datasets_and_ratios.split('_')
    r = [float(x) for x in s_[len(s_) // 2:]]
    d = s_[:len(s_) // 2]
    return d + r


class CheckBatchGradient(pl.Callback):
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)

        for key, out in output.items():
            out[n].abs().sum().backward()

            zero_grad_inds = list(range(example_input.size(0)))
            zero_grad_inds.pop(n)

            if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
                raise RuntimeError(
                    f'Model mixes data across the batch dimension for {key} output!'
                )

            break

        logger.info('Batch gradient test is passed!')


def set_seed(seed_value):
    if seed_value >= 0:
        logger.info(f'Seed value for the experiment {seed_value}')
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        pl.trainer.seed_everything(seed_value)


def add_init_smpl_params_to_dict(state_dict):
    mean_params = np.load(SMPL_MEAN_PARAMS)
    init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
    init_shape = torch.from_numpy(
        mean_params['shape'][:].astype('float32')).unsqueeze(0)
    init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
    state_dict['model.head.init_pose'] = init_pose
    state_dict['model.head.init_shape'] = init_shape
    state_dict['model.head.init_cam'] = init_cam
    return state_dict
