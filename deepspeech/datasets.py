import torch
import torchaudio as audio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deepspeech import utils

# data

def yesno(cfg):
    return audio.datasets.YESNO(
        root=cfg.datasets_dir,
        shuffle=True,
        download=cfg.datasets_dir is not None)


def librispeech(cfg):
    return audio.datasets.LIBRISPEECH(
        root=cfg.datasets_dir,
        shuffle=True,
        download=cfg.datasets_dir is not None)


def commonvoice(cfg, lang='english'):
    return audio.datasets.COMMONVOICE(
        root=cfg.datasets_dir,
        url=lang,
        shuffle=True,
        download=cfg.datasets_dir is not None)


# transforms

def spec_augment(cfg, train=True):
    tt = nn.Sequential(
        audio.transforms.MelSpectrogram(sample_rate=cfg.sampling_rate, n_fft=cfg.n_fft(), n_mels=cfg.n_mels),
        audio.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        audio.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
    )

    if train:
        return tt
    else:
        return tt[0]


# batching

def batch(cfg):
    "numericalize and pad into batch"

    def fn(x, y):
      nx = [len(el) for el in x]
      ny = [len(el) for el in y]
      y = utils.encode_texts(y, cfg.graphemes_idx())
      x = pad_sequence(x, batch_first=True, padding_value=0)
      y = pad_sequence(y, batch_first=True, padding_value=cfg.blank_idx())
      return x, y, nx, ny

    return fn


# datasets

class SpecAugmented(Dataset):

    def __init__(self, data, cfg):
        self.spec_augment = spec_augment(cfg)
        self.data = data
        self.cfg = cfg

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x, y = self.data[0][idx], self.data[1][idx]
        if self.cfg.spec_augmented: x = self.spec_augment(x)
        return x, y

    def __repr__(self):
        return f'SpecAugmented(augmented: { self.cfg.spec_augmented })'
