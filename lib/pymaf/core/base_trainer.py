# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/utils/base_trainer.py
from __future__ import division

import torch
from tqdm import tqdm

tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter

from utils import CheckpointSaver

import logging

logger = logging.getLogger(__name__)


class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        if options.multiprocessing_distributed:
            self.device = torch.device('cuda', options.gpu)
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir,
                                     overwrite=options.overwrite)
        if options.rank == 0:
            self.summary_writer = SummaryWriter(self.options.summary_dir)
        self.init_fn()

        self.checkpoint = None
        if options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(
                self.models_dict, self.optimizers_dict)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

        if self.checkpoint is not None:
            self.checkpoint_batch_idx = self.checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        self.best_performance = float('inf')

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model],
                                                            strict=True)
                    print(f'Checkpoint {model} loaded')

    def move_dict_to_device(self, dict, device, tensor2float=False):
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                if tensor2float:
                    dict[k] = v.float().to(device)
                else:
                    dict[k] = v.to(device)

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def train(self, epoch):
        raise NotImplementedError('You need to provide an train method')

    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError(
            'You need to provide a _train_summaries method')

    def visualize(self, input_batch):
        raise NotImplementedError('You need to provide a visualize method')

    def validate(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        pass

    def fit(self):
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs),
                          total=self.options.num_epochs,
                          initial=self.epoch_count):
            self.epoch_count = epoch
            self.train(epoch)
        return
