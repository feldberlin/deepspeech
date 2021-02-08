import torch

from deepspeech import model, datasets


def test_batch_collation():
    p = model.HParams()
    data = zip(
        [torch.rand(p.n_mels, p.sampling_rate) for x in range(3)],
        ['yes', 'no', 'yes']
    )

    x, nx, y, ny = datasets.batch(p)(data)

    assert x.shape == (3, p.n_mels, p.sampling_rate)
    assert y.shape == (3, 3)
    assert y[1,-1] == p.blank_idx()
    assert nx.equal(torch.tensor([p.sampling_rate] * 3))
    assert ny.equal(torch.tensor([3, 2, 3]))


def test_spec_augmented_dataset():
    p = model.HParams()
    data = zip(
        [torch.rand(p.sampling_rate) for x in range(3)],
        ['yes', 'no', 'yes']
    )

    d = datasets.SpecAugmented(dict(data), p, masked=True)
    assert len(d) == 3
    assert str(d) == 'SpecAugmented()'


def test_splitting():
    p = model.HParams(splits=[0.6, 0.4])
    data = zip(
        [torch.rand(p.sampling_rate) for x in range(3)],
        ['yes', 'no', 'yes']
    )

    subsets = datasets.splits(p, list(data))
    assert len(subsets) == 2
    assert subsets[0].masked == True
    assert subsets[1].masked == False
    for s in subsets:
        assert isinstance(s, datasets.SpecAugmented)
