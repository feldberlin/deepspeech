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


# metrics


class Metrics():

    def __init__(self):
        self.n_word_edits, self.n_words = 0, 0
        self.n_char_edits, self.n_chars = 0, 0

    def __call__(self, x, y):

        # total word edits in batch
        self.n_word_edits += sum([levenshtein(a.split(), b.split())
                            for a, b
                            in zip(x, y)])

        # total char edits in batch
        self.n_char_edits += sum([levenshtein(list(a), list(b))
                            for a, b
                            in zip(x, y)])

        # counts
        self.n_words += sum([l.count(' ') + 1 for l in x])
        self.n_chars += sum([len(l) for l in x])

        return {
            'wer': self.n_word_edits / self.n_words,
            'cer': self.n_char_edits / self.n_chars,
        }


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


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
