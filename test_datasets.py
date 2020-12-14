import torch

from deepspeech import model, datasets


def test_batch():
    p = model.HParams()
    x, y, nx, ny = datasets.batch(p)(
        [torch.rand(p.sampling_rate) for x in range(3)],
        ['yes', 'no', 'yes']
    )

    assert x.shape == (3, p.sampling_rate)
    assert y.shape == (3, 3)
    assert y[1,-1] == p.blank_idx()
    assert nx == [p.sampling_rate] * 3
    assert ny == [3, 2, 3]


def test_dataset():
    p = model.HParams()
    data = (
        [torch.rand(p.sampling_rate) for x in range(3)],
        ['yes', 'no', 'yes']
    )

    d = datasets.SpecAugmented(data, p)
    assert len(d) == 3
    assert str(d) == 'SpecAugmented(augmented: True)'
