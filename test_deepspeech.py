import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from deepspeech import model, utils, datasets

import pytest


def test_hparams():
    p = model.HParams()
    assert p.sampling_rate == 8000
    assert p.n_fft() == 160
    assert p.n_frames() == 101
    assert p.n_downsampled_frames() == 51


def test_hparams_override():
    p = model.HParams(use_mixed_precision=False)
    assert p.use_mixed_precision is False


def test_deepspeech_fwd():
    batch_size = 3
    p = model.HParams()
    m = model.DeepSpeech(p)
    x, y, nx, ny = datasets.batch(p)(
        [torch.rand(p.sampling_rate) for x in range(batch_size)],
        np.random.choice(['yes', 'no'], batch_size)
    )

    x = datasets.spec_augment(p)(x)
    x, _ = m.forward(x, y, nx, ny)

    assert x.shape == (
        p.n_downsampled_frames(),
        batch_size,
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
