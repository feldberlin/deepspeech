import numpy as np
import torch

from deepspeech import model, predict, datasets


def test_ctc_collapse_batch():
    p = model.HParams(graphemes=np.array(['ε', 'a', 'b']))
    assert predict.ctc_collapse_batch(['aababb'], p) == ['abab']
    assert predict.ctc_collapse_batch(['aababεbε'], p) == ['ababb']


def test_greedy():
    p = model.HParams(graphemes=np.array(['ε', 'a', 'b']))
    xs = torch.tensor([[0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.3, 0.2, 0.6]])
    xs = xs.unsqueeze(0)  # B, W, C
    got = predict.decode_argmax(xs, p)
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
    yhat, _ = predict.predict(m, x, xn)
    decoded = predict.decode_argmax(yhat, p)  # make sure we are decodable
    assert len(decoded) == batch_size
