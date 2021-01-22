import torch
import torchaudio as audio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deepspeech import utils

# data

def yesno(cfg):
    return SpecAugmented(YesNo(cfg), cfg, yat=2)


def librispeech(cfg):
    root = cfg.datasets_dir
    data = audio.datasets.LIBRISPEECH(root=root, download=True)
    return SpecAugmented(data, cfg)


def commonvoice(cfg, lang='english'):
    root = cfg.datasets_dir
    data = audio.datasets.COMMONVOICE(root=root, url=lang, download=True)
    return SpecAugmented(data, cfg)


# transforms

def spec_augment(cfg, train=True):
    tt = nn.Sequential(
        audio.transforms.MelSpectrogram(**cfg.mel_config()),
        audio.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        audio.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
    )

    if train:
        return tt
    else:
        return tt[0]


# batching

def batch(cfg):
    "numericalize and pad into a batch"

    def fn(batch):
        x, y = zip(*batch)  # H, W melgrams, strings
        assert x[0].ndim == 2

        # lengths
        nx = torch.tensor([el.shape[1] for el in x])
        ny = torch.tensor([len(el) for el in y])

        # xs
        x = [el.permute(1, 0) for el in x]  # W, H needed for padding
        x = pad_sequence(x, batch_first=True, padding_value=0)
        x = x.permute(0, 2, 1)

        # ys
        y = utils.encode_texts(y, cfg.graphemes_idx())
        y = pad_sequence(y, batch_first=True, padding_value=cfg.blank_idx())

        return x, y, nx, ny

    return fn


# datasets

class SpecAugmented(Dataset):
    """SpecAugmentation as per https://arxiv.org/abs/1904.08779. Warping has
    not been implemented because it has a negligible impact and is not
    implemented in torchaudio.
    """

    def __init__(self, data, cfg, xat=0, yat=1):
        self.spec_augment = spec_augment(cfg)
        self.data = data
        self.cfg = cfg
        self.xat = xat
        self.yat = yat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx][self.xat], self.data[idx][self.yat]
        if self.cfg.spec_augmented: x = self.spec_augment(x)
        return x, y

    def __repr__(self):
        return f'SpecAugmented(augmented: { self.cfg.spec_augmented })'


class YesNo(Dataset):
    """YesNo is a 60 example test dataset. Targets have been converted to
    hebrew text, rather than the original ordinals. Some details of RTL have
    been ignored.
    """

    def __init__(self, cfg):
        self.data = audio.datasets.YESNO(root=cfg.datasets_dir, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        "backwards hebrew yes and no."
        x, fq, y = self.data[idx]
        x = torch.squeeze(x, 0)
        y = ' '.join(['ןכ' if el else 'אל' for el in y])
        return x, fq, y

    def __repr__(self):
        return f'YesNo()'
