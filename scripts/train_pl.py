import sys, os

root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)
from omegaconf import OmegaConf, DictConfig
from render_model.imface_system_finetune import ImfaceSystem
# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

import torch
import yaml
import os
import random
from glob import glob
from easydict import EasyDict as edict

import torch.utils.data
import torch.optim.lr_scheduler
from utils import fileio

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

def main(config, train_config, id2idx, exp2idx):
    setup_seed(config.seed)
    system = ImfaceSystem(config, train_config, id2idx, exp2idx)
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(config.out_dir, 'ckpt', config.expname),
                              save_last=True,
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5,
                              )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir=os.path.join(config.out_dir, "logs"),
                               name=config.expname,
                               default_hp_metric=False)

    trainer = Trainer(
        max_steps=-1,
        max_epochs=config.train.epoch,
        callbacks=callbacks,
        check_val_every_n_epoch=100,
        logger=logger,
        enable_model_summary=False,
        accelerator='auto',
        devices=config.num_gpus,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if config.num_gpus == 1 else None,
        strategy='ddp',
        limit_val_batches=1
    )

    trainer.fit(system, ckpt_path=None)

def load_config(path):
    with open(path, 'r') as f:
        config = f.read()
    config = edict(yaml.load(config, Loader=yaml.FullLoader))
    train_config = glob(os.path.join(config.load_path, '*.yaml'))
    assert len(train_config) == 1
    train_config = train_config[0]
    train_config = OmegaConf.load(train_config)
    train_config = DictConfig(train_config, flags={"allow_objects": True})
    id2idx = fileio.read_dict(os.path.join(config.load_path, 'id2idx.json'))
    exp2idx = fileio.read_dict(os.path.join(config.load_path, 'exp2idx.json'))
    return config, train_config, id2idx, exp2idx

if __name__ == "__main__":
    config_path = os.path.join(root_dir, 'config/train_neuface.yaml')
    config, train_config, id2idx, exp2idx = load_config(config_path)
    print(train_config)
    train_config.DATA_CONFIG.ID_NUM = len(id2idx)
    train_config.DATA_CONFIG.EXP_NUM = len(exp2idx)
    train_config.DATA_CONFIG.TEMPLATE_KPTS = torch.zeros((68, 3)).float()
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '2818'
    main(config, train_config, id2idx, exp2idx)
