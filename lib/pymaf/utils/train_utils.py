import os
import sys
import json
import random
import string
from datetime import datetime
from core.cfgs import cfg

import logging

logger = logging.getLogger(__name__)


def print_args():
    message = ' '.join(sys.argv)
    return message


def prepare_env(args):
    letters = string.ascii_letters
    timestamp = datetime.now().strftime('%b%d-%H-%M-%S-') + ''.join(
        random.choice(letters) for i in range(3))

    sub_dir = 'pymaf_' + cfg.MODEL.PyMAF.BACKBONE
    if not args.single_dataset:
        sub_dir += '_mix'

    # [backbone]_[pretrained datasets]_[aux.supv.]_[loop iteration]_[time]_[random number]
    log_name = sub_dir
    log_name += '_as_' if cfg.MODEL.PyMAF.AUX_SUPV_ON else '_'
    log_name += 'lp' + str(cfg.MODEL.PyMAF.N_ITER)

    if cfg.MODEL.PyMAF.N_ITER > 0:
        log_name += '_mlp'
        log_name += '-'.join(str(i) for i in cfg.MODEL.PyMAF.MLP_DIM)

    log_name += '_' + timestamp
    log_dir = os.path.join(args.log_dir, sub_dir, log_name)

    if not args.resume:
        args.log_name = log_name
        args.log_dir = log_dir
    else:
        args.log_name = args.log_dir.split('/')[-1]

    logger.info('log name: {}'.format(args.log_dir))

    args.summary_dir = os.path.join(args.log_dir, 'tb_summary')
    args.checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')

    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)
    if not os.path.exists(args.checkpoint_dir):
        if args.resume:
            raise ValueError(
                'Experiment are set to resume mode, but checkpoint directory does not exist.'
            )
        os.makedirs(args.checkpoint_dir)

    if not args.resume:
        with open(os.path.join(args.log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        with open(os.path.join(args.log_dir, 'cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
    else:
        with open(os.path.join(args.log_dir, "args_resume.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        with open(os.path.join(args.log_dir, 'cfg_resume.yaml'), 'w') as f:
            f.write(cfg.dump())
