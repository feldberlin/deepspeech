import numpy as np
import torch

from deepspeech import model, predict, datasets


def test_ctc_collapse_batch():
    p = model.HParams(graphemes=np.array(['a', 'b', 'ε']))
    assert predict.ctc_collapse_batch(p, ['aababb']) == ['abab']
    assert predict.ctc_collapse_batch(p, ['aababεbε']) == ['ababb']


def test_greedy():
    p = model.HParams(graphemes=np.array(['a', 'b', 'ε']))
    xs = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6]])
    xs = xs.unsqueeze(0)  # B, W, C
    got = predict.decode_argmax(p, xs)
    assert got == ['ab']


def test_decode_argmax():
    batch_size = 3
    p = model.HParams(mixed_precision=False)
    m = model.DeepSpeech(p)
    data = zip(
        [torch.rand(p.sampling_rate) for x in range(batch_size)],
        ['yes', 'no', 'yes']
    )

    d = datasets.SpecAugmented(list(data), p, masked=True)
    batch = [d[i] for i in range(batch_size)]
    x, xn, y, yn = datasets.batch(p)(batch)
    yhat = predict.predict(m, x, xn)
    decoded = predict.decode_argmax(p, yhat)  # make sure we are decodable
    assert len(decoded) == batch_size
