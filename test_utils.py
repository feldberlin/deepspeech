import torch
import numpy as np

from deepspeech import utils, model, train


# data

def test_encode_text():
    got = utils.encode_text('definitely mayb', model.HParams().graphemes_idx())
    want = np.array([3, 4, 5, 8, 13, 8, 19, 4, 11, 24, 26, 12, 0, 24, 1])
    assert np.array_equal(got, want)


def test_encode_texts():
    got = utils.encode_texts(['a', 'b'], model.HParams().graphemes_idx())
    assert np.array_equal(
        np.array(got),
        np.array([torch.tensor([0]), torch.tensor([1])])
    )


def test_decode_text():
    encoded = np.array([3, 4, 5, 8, 13, 8, 19, 4, 11, 24, 26, 12, 0, 24, 1])
    assert utils.decode_text(encoded, model.HParams()) == 'definitely mayb'


def test_decode_texts():
    got = utils.decode_texts(np.array([[0], [1]]), model.HParams())
    assert np.array_equal(got, np.array(['a', 'b']))


# config

def test_hparams_dict():
    class TestHParams(utils.HParams):
        a = 'b'
        foo = 'bar'

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    p = TestHParams(foo='DELETED', baz='qux')
    assert dict(p) == {'a': 'b', 'foo': 'DELETED', 'baz': 'qux'}


def test_load_hparams():
    with open('fixtures/config.yaml', 'r') as f:
        p, ptrain = utils.load_hparams(f)
    assert not p.get('train', None)
    assert ptrain['batch_size'] == 64


# schedules

def test_lrfinder():
    m = model.DeepSpeech(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-8)
    p = train.HParams(batch_size=1, max_epochs=1)
    schedule = utils.lrfinder(optimizer, 9, p)
    assert np.isclose(schedule.gamma, 10.)


def test_onecycle():
    cfg = train.HParams(batch_size=1, max_epochs=1)
    m = model.DeepSpeech(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate)
    schedule = utils.onecycle(optimizer, 9, cfg)
    assert schedule.total_steps == 9
