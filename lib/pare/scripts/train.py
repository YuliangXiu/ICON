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

import comet_ml

import os
import sys
import torch
import random
import pprint
import argparse
import subprocess
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

sys.path.append('')

from pare.utils.os_utils import copy_code
from pare.utils.eval_utils import find_best_ckpt
from pare.core.trainer import PARETrainer
from pare.core.config import run_grid_search_experiments
from pare.utils.train_utils import load_pretrained_model, resume_training, set_seed, \
    add_init_smpl_params_to_dict, CheckBatchGradient


def main(hparams, disable_comet=False, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    copy_code(output_folder=log_dir,
              curr_folder=os.path.dirname(os.path.abspath(__file__)))

    logger.info(torch.cuda.get_device_properties(device))
    hparams.SYSTEM.GPU = torch.cuda.get_device_properties(device).name

    logger.info(f'Hyperparameters: \n {hparams}')

    experiment_loggers = []

    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tb_logs',
        log_graph=False,
    )

    experiment_loggers.append(tb_logger)

    model = PARETrainer(hparams=hparams).to(device)

    # TRAINING.PRETRAINED_LIT points to the checkpoint files trained using this repo
    # This has a separate cfg value since in some cases we use checkpoint files from different repos
    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(
            f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    if hparams.TRAINING.RESUME is not None:
        resume_ckpt = torch.load(hparams.TRAINING.RESUME)
        if not 'model.head.init_pose' in resume_ckpt['state_dict'].keys():
            logger.info(
                'Adding init SMPL parameters to the resume checkpoint...')
            resume_ckpt = torch.load(hparams.TRAINING.RESUME)
            resume_ckpt['state_dict'] = add_init_smpl_params_to_dict(
                resume_ckpt['state_dict'])
            torch.save(resume_ckpt, hparams.TRAINING.RESUME)

    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=30,
        mode='min',
        period=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
    )

    amp_params = {}
    if hparams.TRAINING.USE_AMP:
        logger.info(
            f'Using automatic mixed precision: ampl_level 02, precision 16...')
        amp_params = {
            'amp_level': 'O2',
            # 'amp_backend': 'apex',
            'precision': 16,
        }

    trainer = pl.Trainer(
        gpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[ckpt_callback],
        log_every_n_steps=50,
        terminate_on_nan=True,
        default_root_dir=log_dir,
        progress_bar_refresh_rate=50,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        # checkpoint_callback=ckpt_callback,
        reload_dataloaders_every_epoch=hparams.TRAINING.
        RELOAD_DATALOADERS_EVERY_EPOCH,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        **amp_params,
    )

    if hparams.TRAINING.TEST_BEFORE_TRAINING:
        logger.info(
            f'Running an initial on {hparams.DATASET.VAL_DS} test before training'
        )
        trainer.test(model)

    logger.info('*** Started training ***')
    trainer.fit(model)
    # trainer.test(model)
    if hparams.TESTING.TEST_ON_TRAIN_END:
        ckpt_path = find_best_ckpt(os.path.join(hparams.LOG_DIR,
                                                'config_to_run.yaml'),
                                   new_version=True)

        logger.info('*** Started testing on all datasets ***')
        for val_ds in ['3doh', 'mpi-inf-3dhp', '3dpw-all']:
            cmd = [
                'python', 'eval.py', '--cfg',
                os.path.join(hparams.LOG_DIR, 'config_to_run.yaml'), '--opts',
                'DATASET.VAL_DS', val_ds, 'TRAINING.PRETRAINED_LIT', ckpt_path
            ]
            subprocess.call(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--opts',
                        default=[],
                        nargs='*',
                        help='additional options to update config')
    parser.add_argument(
        '--cfg_id',
        type=int,
        default=0,
        help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster',
                        default=False,
                        action='store_true',
                        help='creates submission files for cluster')
    parser.add_argument('--resume',
                        default=False,
                        action='store_true',
                        help='resume training from where it left off')
    parser.add_argument(
        '--resume_wo_optimizer',
        default=False,
        action='store_true',
        help='resume training from where it left off but do not use optimizer')
    parser.add_argument('--bid',
                        type=int,
                        default=5,
                        help='amount of bid for cluster')
    parser.add_argument('--memory',
                        type=int,
                        default=64000,
                        help='memory amount for cluster')
    parser.add_argument('--num_cpus',
                        type=int,
                        default=8,
                        help='num cpus for cluster')
    parser.add_argument('--gpu_min_mem',
                        type=int,
                        default=10000,
                        help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch',
                        default=['tesla', 'quadro', 'rtx'],
                        nargs='*',
                        help='additional options to update config')
    parser.add_argument('--disable_comet', action='store_true')
    parser.add_argument('--fdr', action='store_true')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    if args.resume:
        resume_training(args)

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='train.py',
        cmd_opts=args.opts,
        gpu_min_mem=args.gpu_min_mem,
        gpu_arch=args.gpu_arch,
    )

    main(hparams, disable_comet=args.disable_comet, fast_dev_run=args.fdr)
