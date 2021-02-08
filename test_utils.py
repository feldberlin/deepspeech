import torch

from deepspeech import utils, model, train


# data

def test_encode_text():
    got = utils.encode_text('definitely mayb', model.HParams().graphemes_idx())
    want = [3, 4, 5, 8, 13, 8, 19, 4, 11, 24, 26, 12, 0, 24, 1]
    assert got.tolist() == want


def test_encode_texts():
    got = utils.encode_texts(['aa', 'bb'], model.HParams().graphemes_idx())
    want = [torch.tensor([0, 0]), torch.tensor([1, 1])]
    assert torch.stack(got).equal(torch.stack(want))


def test_decode_text():
    encoded = [3, 4, 5, 8, 13, 8, 19, 4, 11, 24, 26, 12, 0, 24, 1]
    assert utils.decode_text(encoded, model.HParams()) == 'definitely mayb'


def test_decode_texts():
    got = utils.decode_texts([[0, 1], [1, 0]], model.HParams())
    assert got == ['ab', 'ba']


# metrics


def test_metrics():
    x = ['this here is a tricky sentence', 'the lights are green']
    y = ['this hore is tricky a sentence', 'the lights are red']

    m = utils.Metrics()
    m.accumulate(x, y)

    assert m.to_dict() == {
        'wer': (3 + 1) / (6 + 4),
        'cer': (5 + 3) / (30 + 20),
    }


def test_metrics_accumulation():
    x = ['this here is a tricky sentence', 'the lights are green']
    y = ['this hore is tricky a sentence', 'the lights are red']

    m = utils.Metrics()
    m.accumulate(x[:1], y[:1])
    m.accumulate(x[1:], y[1:])

    assert m.to_dict() == {
        'wer': (3 + 1) / (6 + 4),
        'cer': (5 + 3) / (30 + 20),
    }


def test_levenshtein():
    a = 'this here is a tricky sentence'
    b = 'this hore is tricky a sentence'
    assert utils.levenshtein(a.split(), b.split()) == 3


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


def test_hparams():
    p = model.HParams()
    assert not dict(p).get('graphemes_idx', None)


def test_load_hparams():
    p, ptrain = utils.load_hparams('./fixtures/config.yaml')
    assert not p.get('train', None)
    assert ptrain['batch_size'] == 64


# schedules

def test_lrfinder():
    m = model.DeepSpeech(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-8)
    p = train.HParams(batch_size=1, max_epochs=1)
    schedule = utils.lrfinder(optimizer, 9, p)
    assert torch.isclose(torch.tensor(schedule.gamma), torch.tensor(10.))


def test_onecycle():
    cfg = train.HParams(batch_size=1, max_epochs=1)
    m = model.DeepSpeech(model.HParams())
    optimizer = torch.optim.SGD(m.parameters(), lr=cfg.learning_rate)
    schedule = utils.onecycle(optimizer, 9, cfg)
    assert schedule.total_steps == 9
