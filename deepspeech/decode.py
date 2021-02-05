import itertools

import torch

from deepspeech import utils


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
