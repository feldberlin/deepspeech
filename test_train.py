from deepspeech import train, utils


def test_hparams_nsteps():
    trainset_size = 80
    tp = train.HParams(batch_size=2, max_epochs=10)
    assert tp.n_steps(trainset_size) == (80 / 2) * 10


def test_hparams_nsteps_batch_too_large():
    trainset_size = 80
    tp = train.HParams(batch_size=80, max_epochs=10)
    assert tp.n_steps(trainset_size) == (80 / 80) * 10


def test_hparams_nsteps_last_batch_small():
    trainset_size = 48
    tp = train.HParams(batch_size=40, max_epochs=4)
    assert tp.n_steps(trainset_size) == 8
