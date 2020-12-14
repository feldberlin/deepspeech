import inspect
import yaml

import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import wandb


# data

def encode_text(text, idx):
    return torch.tensor([idx[c] for c in text])


def encode_texts(texts, idx):
    return [encode_text(t, idx) for t in texts]


def decode_text(code, cfg):
    return ''.join(cfg.graphemes[code])


def decode_texts(codes, cfg):
    return [decode_text(c, cfg) for c in codes]


# schedules

def lrfinder(optimizer, n_examples, cfg):
    start_lr, final_lr = 1e-8, 10.
    n_steps = cfg.n_steps(n_examples)
    gamma = (final_lr / start_lr) ** (1/n_steps)
    return lr_scheduler.ExponentialLR(optimizer, gamma)


def onecycle(optimizer, n_examples, cfg):
    lr = cfg.learning_rate
    n_steps = cfg.n_steps(n_examples)
    return lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_steps)


# lifecycle


def load_chkpt(m, run_path):
    chkpt = wandb.restore('checkpoints.best.test', run_path=run_path)
    m.load_state_dict(torch.load(chkpt.name))
    return m


# config

def cfgdict(model_cfg, train_cfg):
    return {**dict(model_cfg), 'train': dict(train_cfg)}


class HParams():
    "Make HParams iterable so we can call dict on it"
    def __iter__(self):
        def f(obj):
            return {k: v for k, v
                    in vars(obj).items()
                    if not k.startswith('__')
                    and not inspect.isfunction(v)}

        return iter({**f(self.__class__), **f(self)}.items())


def load_hparams(path):
    "Load model, train cfgs from wandb formatted yaml"
    p = yaml.safe_load(path)
    del p['_wandb']
    del p['wandb_version']
    return (
        { k: v['value'] for k, v in p.items() if k != 'train' },
        p.pop('train')['value']
    )


def load_wandb_cfg(run_path):
    "Load model and train cfg from wandb"
    return load_hparams(wandb.restore('config.yaml', run_path=run_path))
