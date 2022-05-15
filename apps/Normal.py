import logging
import os
import sys
import torch
import numpy as np
from PIL import Image
from torch import nn
import os.path as osp
from skimage.transform import resize
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True

logging.getLogger("lightning").setLevel(logging.ERROR)

sys.path.insert(0, '../')

from lib.common.train_util import *
from lib.net import NormalNet


class Normal(pl.LightningModule):

    def __init__(self, cfg):
        super(Normal, self).__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_N = self.cfg.lr_N
        
        self.schedulers = []

        self.netG = NormalNet(self.cfg,
                              error_term=nn.SmoothL1Loss())
        
        self.in_nml = [item[0] for item in cfg.net.in_nml]


    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum
            
        optim_params_N_F = [{'params': self.netG.netF.parameters(), 'lr': self.lr_N}]
        optim_params_N_B = [{'params': self.netG.netB.parameters(), 'lr': self.lr_N}]

        optimizer_N_F = torch.optim.Adam(
                    optim_params_N_F,
                    lr=self.lr_N, 
                    weight_decay=weight_decay)
                
        optimizer_N_B = torch.optim.Adam(
                    optim_params_N_B,
                    lr=self.lr_N, 
                    weight_decay=weight_decay)
    
        scheduler_N_F = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_F, milestones=self.cfg.schedule, gamma=self.cfg.gamma)
        
        scheduler_N_B = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_B, milestones=self.cfg.schedule, gamma=self.cfg.gamma)
        
        self.schedulers = [scheduler_N_F, scheduler_N_B]
        optims = [optimizer_N_F, optimizer_N_B]
            
        return optims, self.schedulers
    
    def render_func(self, render_tensor):
        
        height = render_tensor['image'].shape[2]
        result_list = []
        
        for name in render_tensor.keys():
            result_list.append(
                        resize(((render_tensor[name].cpu().numpy()[0]+1.0)/2.0).transpose(1, 2, 0),
                        (height, height), anti_aliasing=True))
        result_array = np.concatenate(result_list, axis=1)
        
        return result_array
        

    def training_step(self, batch, batch_idx, optimizer_idx):

        export_cfg(self.logger, self.cfg)

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]
            
        FB_tensor = {'normal_F': batch['normal_F'],
                     'normal_B': batch['normal_B']}
          
        self.netG.train()
        
        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B, FB_tensor)
  
        (opt_nf, opt_nb) = self.optimizers()

        opt_nf.zero_grad()
        opt_nb.zero_grad()
        
        self.manual_backward(error_NF, opt_nf)
        self.manual_backward(error_NB, opt_nb)
        
        opt_nf.step()
        opt_nb.step()
        
        if batch_idx > 0 and batch_idx % int(self.cfg.freq_show_train) == 0:
            
            self.netG.eval()
            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF,
                                "nmlB": nmlB})
                result_array = self.render_func(in_tensor)

                self.logger.experiment.add_image(
                    tag=f'Normal-train/{self.global_step}',
                    img_tensor=result_array.transpose(2, 0, 1),
                    global_step=self.global_step)
            
        # metrics processing
        metrics_log = {"train_loss-NF": error_NF.item(),
                       "train_loss-NB": error_NB.item()}
        
        tf_log = tf_log_convert(metrics_log)
        bar_log = bar_log_convert(metrics_log)

        return {'loss': error_NF + error_NB,
                'loss-NF': error_NF,
                'loss-NB': error_NB,
                'log': tf_log,
                'progress_bar': bar_log}

    def training_epoch_end(self, outputs):

        if [] in outputs:
            outputs = outputs[0]

        # metrics processing
        metrics_log = {"train_avgloss": batch_mean(outputs, 'loss'),
                       "train_avgloss-NF": batch_mean(outputs, 'loss-NF'),
                       "train_avgloss-NB": batch_mean(outputs, 'loss-NB')}

        tf_log = tf_log_convert(metrics_log)
     
        tf_log['lr-NF'] = self.schedulers[0].get_last_lr()[0]
        tf_log['lr-NB'] = self.schedulers[1].get_last_lr()[0]

        return {'log': tf_log}

    def validation_step(self, batch, batch_idx):

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]
            
        FB_tensor = {'normal_F': batch['normal_F'],
                     'normal_B': batch['normal_B']}
          
        self.netG.train()
        
        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B, FB_tensor)
        
        if (batch_idx > 0 and batch_idx % int(self.cfg.freq_show_train) == 0) or (batch_idx==0):
            
            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF,
                                "nmlB": nmlB})
                result_array = self.render_func(in_tensor)

                self.logger.experiment.add_image(
                    tag=f'Normal-val/{self.global_step}',
                    img_tensor=result_array.transpose(2, 0, 1),
                    global_step=self.global_step)

     
        return {"val_loss": error_NF + error_NB,
                "val_loss-NF": error_NF,
                "val_loss-NB": error_NB}

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = { "val_avgloss": batch_mean(outputs, 'val_loss'),
                        "val_avgloss-NF": batch_mean(outputs, 'val_loss-NF'),
                        "val_avgloss-NB": batch_mean(outputs, 'val_loss-NB')}

        tf_log = tf_log_convert(metrics_log)
        
        return {'log': tf_log}
