import sys
import os
import os.path as osp
import argparse
import torch
import numpy as np

# project related libs
sys.path.insert(0, '../')
from lib.common.config import get_cfg_defaults

# pytorch lightning related libs
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from apps.Normal import Normal
from lib.dataset.NormalModule import NormalModule
from pytorch_lightning.callbacks import LearningRateMonitor

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', type=str, help='path of the yaml config file')
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.ckpt_dir, cfg.name), exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.results_path,
                                            name=cfg.name,
                                            default_hp_metric=False)
    
    save_k=3
    
    if cfg.overfit:
        cfg_overfit_list = ['batch_size', 1]
        cfg.merge_from_list(cfg_overfit_list)
        save_k = 0
    
    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=save_k,
        verbose=False,
        save_weights_only=True,
        monitor='val/avgloss',
        mode='min',
        filename="Normal-{epoch:02d}"
    )

    freq_eval = cfg.freq_eval
    if cfg.fast_dev > 0:
        freq_eval = cfg.fast_dev

    trainer_kwargs = {
        # 'accelerator': 'dp',
        # 'amp_level': 'O2',
        # 'precision': 16,
        # 'weights_summary': 'top',
        # 'stochastic_weight_avg': False,
        'gpus': cfg.gpus,
        'auto_select_gpus': True,
        'reload_dataloaders_every_epoch': True,
        'sync_batchnorm': True,
        'benchmark': True,
        'automatic_optimization': False,
        'logger':tb_logger,
        'track_grad_norm': -1,
        'num_sanity_val_steps': cfg.num_sanity_val_steps,
        'checkpoint_callback':checkpoint,
        'limit_train_batches':cfg.dataset.train_bsize,
        'limit_val_batches':cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        'limit_test_batches':cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        'profiler': None,
        'fast_dev_run':cfg.fast_dev,
        'max_epochs':cfg.num_epoch,
        'callbacks': [LearningRateMonitor(logging_interval='step')]
    }

    datamodule = NormalModule(cfg)


    if not cfg.test_mode:
        datamodule.setup(stage='fit')
        train_len = datamodule.data_size['train']
        val_len = datamodule.data_size['val']
        trainer_kwargs.update(
            {'log_every_n_steps': int(cfg.freq_plot * train_len / cfg.batch_size),
            'val_check_interval': int(freq_eval * train_len / cfg.batch_size) if freq_eval > 10 else freq_eval}
        )
        if cfg.overfit:
            cfg_show_list = ['freq_show_train', 200.0,
                             'freq_show_val', 10.0]
        else:
            cfg_show_list = ['freq_show_train', cfg.freq_show_train * train_len / cfg.batch_size,
                        'freq_show_val', max(cfg.freq_show_val * val_len / cfg.batch_size, 1.0)]
                        
        cfg.merge_from_list(cfg_show_list)
        
    model = Normal(cfg)

    trainer = pl.Trainer(**trainer_kwargs)

    if cfg.resume and os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
        trainer_kwargs['resume_from_checkpoint'] = cfg.resume_path
        trainer = pl.Trainer(**trainer_kwargs)
        print(f"Resume weights and hparams from {cfg.resume_path}")
    elif not cfg.resume and os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):

        pretrained_dict = torch.load(cfg.resume_path, map_location=torch.device(f"cuda:{cfg.gpus[0]}"))['state_dict']
        model_dict = model.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and v.shape == model_dict[k].shape}
      
        # # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        
        del pretrained_dict
        del model_dict

        print(f"Resume only weights from {cfg.resume_path}")
    else:
        pass

    if not cfg.test_mode:

        trainer.fit(model=model, datamodule=datamodule)
        
    else:
        np.random.seed(1993)
        trainer.test(model=model, datamodule=datamodule)
