import os
import pytest

import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from deepspeech import model, utils, datasets, train


def test_hparams():
    p = model.HParams()
    assert p.sampling_rate == 8000
    assert p.n_fft() == 160
    assert p.n_downsampled_frames(101) == 51


def test_hparams_override():
    p = model.HParams(use_mixed_precision=False)
    assert p.use_mixed_precision is False


def test_hparams_blank():
    p = model.HParams()
    assert p.graphemes[0] == 'ε'


def test_hparams_graphemes_idx():
    p = model.HParams(graphemes=datasets.YESNO_GRAPHEMES)
    assert p.graphemes_idx() == {
        'ε': 0,
        'e': 1,
        'k': 2,
        'l': 3,
        'n': 4,
        'o': 5,
        'r': 6,
        ' ': 7,
    }


def test_deepspeech_fwd():
    batch_size = 5
    p = model.HParams()
    transform = datasets.transform(p)
    m = model.DeepSpeech(p)

    # follow the same order as in data loader and trainer
    x = [transform(torch.rand(p.sampling_rate)) for x in range(batch_size)]
    y = np.random.choice(['yes', 'no'], batch_size)
    x, nx, y, ny = datasets.batch(p)(zip(x, y))

    "need: (B, H, W) batches of melspecs, (B, W) batches of graphemes."
    print(x.shape, nx, y.shape, ny)

    x, _ = m.forward(x, nx, y, ny)

    assert x.shape == (
        batch_size,
        49,
        p.n_graphemes()
    )


def test_deepspeech_fwd_augmented():
    batch_size = 5
    p = model.HParams()
    transform = datasets.transform(p)
    augment = datasets.spec_augment(p)
    m = model.DeepSpeech(p)

    # follow the same order as in data loader and trainer
    x = [augment(transform(torch.rand(p.sampling_rate))) for x in range(batch_size)]
    y = np.random.choice(['yes', 'no'], batch_size)
    x, nx, y, ny = datasets.batch(p)(zip(x, y))
    x, _ = m.forward(x, nx, y, ny)

    assert x.shape == (
        batch_size,
        49,
        p.n_graphemes()
    )


def test_deepspeech_modules_registered():
    m = model.DeepSpeech(model.HParams(n_layers=1, dilation_stacks=1))
    got = list(m.state_dict().keys())
    want = [
        'conv.weight',
        'conv.bias',
        'dense_a.weight',
        'dense_a.bias',
        'dense_b.weight',
        'dense_b.bias',
        'gru.weight_ih_l0',
        'gru.weight_hh_l0',
        'gru.bias_ih_l0',
        'gru.bias_hh_l0',
        'gru.weight_ih_l0_reverse',
        'gru.weight_hh_l0_reverse',
        'gru.bias_ih_l0_reverse',
        'gru.bias_hh_l0_reverse',
        'dense_end.weight',
        'dense_end.bias'
    ]

    assert got == want


@pytest.mark.integration
def test_deepspeech_train():

    # do not call home to wandb
    os.environ['WANDB_MODE'] = 'dryrun'

    # hyperparams
    p = model.HParams(graphemes=datasets.YESNO_GRAPHEMES)
    tp = train.HParams(max_epochs=1, batch_size=8)

    # build
    m = model.DeepSpeech(p)
    trainset, testset = datasets.splits(datasets.YesNo(p), p)

    # train
    t = train.Trainer(m, trainset, testset, tp)
    t.train()
