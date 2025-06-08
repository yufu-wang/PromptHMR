import argparse
# import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.vis.rich_logger import print_cfg
from hmr4d.utils.net_utils import load_pretrained_model, get_resume_ckpt_path
from hmr4d.datamodule.mocap_trainX_testY import DataModule  
from hmr4d.model.gvhmr.gvhmr_pl import GvhmrPL
from hmr4d.utils.callbacks.simple_ckpt_saver import SimpleCkptSaver
from hmr4d.utils.callbacks.prog_bar import ProgressReporter
from hmr4d.utils.callbacks.train_speed_timer import TrainSpeedTimer
from hmr4d.model.gvhmr.callbacks.metric_emdb import MetricMocap as MetricEmdb
from hmr4d.model.gvhmr.callbacks.metric_rich import MetricMocap as MetricRich
from hmr4d.model.gvhmr.callbacks.metric_3dpw import MetricMocap as Metric3dpw

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def get_callbacks(cfg: DictConfig) -> list:
    """Parse and instantiate all the callbacks in the config."""
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    # Handle special callbacks
    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    # Instantiate all the callbacks
    callbacks = []
    callbacks.append(
        SimpleCkptSaver(
            output_dir=cfg.output_dir + "/checkpoints",
            filename=cfg.callbacks.simple_ckpt_saver.filename,
            save_top_k=cfg.callbacks.simple_ckpt_saver.save_top_k,
            every_n_epochs=cfg.callbacks.simple_ckpt_saver.every_n_epochs,
            save_last=cfg.callbacks.simple_ckpt_saver.save_last,
            save_weights_only=cfg.callbacks.simple_ckpt_saver.save_weights_only,
        )
    )
    callbacks.append(
        ProgressReporter(
            log_every_percent=cfg.callbacks.prog_bar.log_every_percent,
            exp_name=cfg.exp_name_base + cfg.exp_name_var,
            data_name=cfg.data_name,
        )
    )
    callbacks.append(TrainSpeedTimer(N_avg=cfg.callbacks.train_speed_timer.N_avg))
    callbacks.append(LearningRateMonitor())
    callbacks.append(MetricEmdb(emdb_split=1))
    callbacks.append(MetricEmdb(emdb_split=2))
    callbacks.append(MetricRich())
    callbacks.append(Metric3dpw())
    return callbacks


def train(cfg: DictConfig) -> None:
    """Train/Test"""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    pl.seed_everything(cfg.seed)

    model = GvhmrPL(
        pipeline=cfg.model.pipeline,
        optimizer=cfg.model.optimizer,
        scheduler_cfg=cfg.model.scheduler_cfg,
        ignored_weights_prefix=cfg.model.ignored_weights_prefix,
    )
    
    datamodule = DataModule(
        dataset_opts=cfg.data.dataset_opts,
        loader_opts=cfg.data.loader_opts,
        limit_each_trainset=cfg.data.limit_each_trainset,
        task=cfg.task,
    )
    
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path)

    # PL callbacks and logger
    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any([isinstance(cb, Checkpoint) for cb in callbacks])
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}

    if cfg.task == 'fit':
        logger = TensorBoardLogger(
            save_dir=cfg.logger.save_dir, 
            name=cfg.logger.name, 
            version=cfg.logger.version
        )
    else:
        logger = None

    # PL-Trainer
    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=cfg.callbacks.model_checkpoint.dirpath)
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start Fitiing...")
        trainer.test(model, datamodule.test_dataloader())
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start Testing...")
        trainer.test(model, datamodule.test_dataloader())
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    Log.info("End of script.")


def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    register_store_gvhmr()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    cfg_file = OmegaConf.load(args.cfg_file)
    cfg = OmegaConf.merge(cfg_file, OmegaConf.from_cli(extras))
    
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    
    main(cfg)
