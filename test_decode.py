import torch
import numpy as np

from deepspeech import decode, model


def test_ctc_collapse_batch():
    p = model.HParams(graphemes=np.array(['a', 'b', 'ε']))
    assert decode.ctc_collapse_batch(p, ['aababb']) == ['abab']
    assert decode.ctc_collapse_batch(p, ['aababεbε']) == ['ababb']


def test_greedy():
    p = model.HParams(graphemes=np.array(['a', 'b', 'ε']))
    xs = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6]])
    xs = xs.unsqueeze(0)  # B, W, C
    got = decode.decode_argmax(p, xs)
    assert got == ['ab']
