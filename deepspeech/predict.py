import itertools

import torch
import torch.cuda.amp as amp

from deepspeech import model, train, utils


def predict(m, x, nx):
    "run a forward pass on a single batch"

    # cpu and gpu both work
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    # run inference
    with amp.autocast(enabled=m.cfg.mixed_precision):
        m = m.to(device)
        x = x.to(device)
        nx = nx.to(device)
        with torch.set_grad_enabled(False):
            yhat, _ = m.forward(x, nx)
        return yhat


# ctc utilities

def ctc_collapse(cfg, x):
    x = ''.join(c for c, _ in itertools.groupby(x))
    return x.replace(cfg.graphemes[-1], '')


def ctc_collapse_batch(cfg, xs):
    return [ctc_collapse(cfg, x) for x in xs]


# decoding strategies

def decode_argmax(cfg, x):
    "(B, W, C) probabilities."

    x = torch.argmax(x, dim=2).tolist()
    x = utils.decode_texts(x, cfg)
    return ctc_collapse_batch(cfg, x)


# utils

def load(run_path):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.DeepSpeech(p), run_path), ptrain
