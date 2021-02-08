import itertools

import torch
import torch.cuda.amp as amp

from deepspeech import model, train, utils


def predict(m, x, nx, y=None, yn=None):
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
            return m.forward(x, nx, y, yn)


# ctc utilities

def ctc_collapse(x, cfg):
    x = ''.join(c for c, _ in itertools.groupby(x))
    return x.replace(cfg.graphemes[-1], '')


def ctc_collapse_batch(xs, cfg):
    return [ctc_collapse(x, cfg) for x in xs]


# decoding strategies

def decode_argmax(x, cfg):
    "(B, W, C) probabilities."

    x = torch.argmax(x, dim=2).tolist()
    x = utils.decode_texts(x, cfg)
    return ctc_collapse_batch(x, cfg)


# utils

def load(run_path):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.DeepSpeech(p), run_path), ptrain
