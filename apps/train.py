import logging
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from termcolor import colored
from apps.ICON import ICON
from lib.dataset.PIFuDataModule import PIFuDataModule
from lib.common.config import get_cfg_defaults
import os
import os.path as osp
import argparse
import torch
import numpy as np

logging.getLogger("lightning").setLevel(logging.ERROR)


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


class SubTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        """Save model/training states as a checkpoint file through state-dump and file-write.
        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        _checkpoint = self.checkpoint_connector.dump_checkpoint(weights_only)

        del_keys = []
        for key in _checkpoint["state_dict"].keys():
            for ig_key in ["normal_filter", "voxelization", "reconEngine"]:
                if ig_key in key:
                    del_keys.append(key)
        for key in del_keys:
            del _checkpoint["state_dict"][key]

        if self.is_global_zero:
            # write the checkpoint dictionary on the file

            if self.training_type_plugin:
                checkpoint = self.training_type_plugin.on_save(_checkpoint)
            try:
                atomic_save(checkpoint, filepath)
            except AttributeError as err:
                if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                    del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                rank_zero_warn(
                    "Warning, `hyper_parameters` dropped from checkpoint."
                    f" An attribute is not picklable {err}"
                )
                atomic_save(checkpoint, filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config_file", type=str, help="path of the yaml config file"
    )
    parser.add_argument("-test", "--test_mode", action="store_true")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.ckpt_dir, cfg.name), exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.results_path, name=cfg.name, default_hp_metric=False
    )

    save_k = 3

    if cfg.overfit:
        cfg_overfit_list = ["batch_size", 1]
        cfg.merge_from_list(cfg_overfit_list)
        save_k = 0

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=save_k,
        verbose=False,
        save_weights_only=True,
        monitor="val/avgloss",
        mode="min",
        filename="{epoch:02d}",
    )

    if cfg.test_mode or args.test_mode:

        cfg_test_mode = [
            "test_mode",
            True,
            "dataset.types",
            ["renderpeople", "cape"],
            "dataset.rp_type",
            "pifu450",
            "dataset.scales",
            [1.0, 100.0],
        ]
        cfg.merge_from_list(cfg_test_mode)

    freq_eval = cfg.freq_eval
    if cfg.fast_dev > 0:
        freq_eval = cfg.fast_dev

    trainer_kwargs = {
        "gpus": cfg.gpus,
        "auto_select_gpus": True,
        "reload_dataloaders_every_epoch": True,
        "sync_batchnorm": True,
        "benchmark": True,
        "logger": tb_logger,
        "track_grad_norm": -1,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "checkpoint_callback": checkpoint,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "profiler": None,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [LearningRateMonitor(logging_interval="step")],
    }

    datamodule = PIFuDataModule(cfg)

    if not cfg.test_mode:
        datamodule.setup(stage="fit")
        train_len = datamodule.data_size["train"]
        val_len = datamodule.data_size["val"]
        trainer_kwargs.update(
            {
                "log_every_n_steps": int(cfg.freq_plot * train_len / cfg.batch_size),
                "val_check_interval": int(freq_eval * train_len / cfg.batch_size)
                if freq_eval > 10
                else freq_eval,
            }
        )

        if cfg.overfit:
            cfg_show_list = ["freq_show_train", 100.0, "freq_show_val", 10.0]
        else:
            cfg_show_list = [
                "freq_show_train",
                cfg.freq_show_train * train_len // cfg.batch_size,
                "freq_show_val",
                max(cfg.freq_show_val * val_len, 1.0),
            ]

        cfg.merge_from_list(cfg_show_list)

    model = ICON(cfg)

    trainer = SubTrainer(**trainer_kwargs)

    if (
        cfg.resume
        and os.path.exists(cfg.resume_path)
        and cfg.resume_path.endswith("ckpt")
    ):

        trainer_kwargs["resume_from_checkpoint"] = cfg.resume_path
        trainer = SubTrainer(**trainer_kwargs)
        print(
            colored(f"Resume weights and hparams from {cfg.resume_path}", "green"))

    elif not cfg.resume:

        model_dict = model.state_dict()
        main_dict = {}
        normal_dict = {}

        if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
            main_dict = torch.load(
                cfg.resume_path, map_location=torch.device(
                    f"cuda:{cfg.gpus[0]}")
            )["state_dict"]

            main_dict = {
                k: v
                for k, v in main_dict.items()
                if k in model_dict
                and v.shape == model_dict[k].shape
                and ("reconEngine" not in k)
                and ("normal_filter" not in k)
                and ("voxelization" not in k)
            }
            print(
                colored(f"Resume MLP weights from {cfg.resume_path}", "green"))

        if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
            normal_dict = torch.load(
                cfg.normal_path, map_location=torch.device(
                    f"cuda:{cfg.gpus[0]}")
            )["state_dict"]

            for key in normal_dict.keys():
                normal_dict = rename(
                    normal_dict, key, key.replace("netG", "netG.normal_filter")
                )

            normal_dict = {
                k: v
                for k, v in normal_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            print(
                colored(f"Resume normal model from {cfg.normal_path}", "green"))

        model_dict.update(main_dict)
        model_dict.update(normal_dict)
        model.load_state_dict(model_dict)

        del main_dict
        del normal_dict
        del model_dict

    else:
        pass

    if not cfg.test_mode:
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
    else:
        np.random.seed(1993)
        trainer.test(model=model, datamodule=datamodule)
