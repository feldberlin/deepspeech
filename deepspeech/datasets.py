import torch
import torchaudio as ta
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deepspeech import utils

# data

def yesno(cfg):
    return SpecAugmented(YesNo(cfg), cfg)


def librispeech(cfg):
    root = cfg.datasets_dir
    data = ta.datasets.LIBRISPEECH(root=root, download=True)
    return SpecAugmented(data, cfg)


def commonvoice(cfg, lang='english'):
    root = cfg.datasets_dir
    data = ta.datasets.COMMONVOICE(root=root, url=lang, download=True)
    return SpecAugmented(data, cfg)


# transforms

def spec_augment(cfg, train=True):
    tt = nn.Sequential(
        ta.transforms.MelSpectrogram(**cfg.mel_config()),
        ta.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        ta.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
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

        # lengths. nx in mel frames
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

    def __init__(self, data, cfg):
        self.spec_augment = spec_augment(cfg)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return self.spec_augment(x), y

    def __repr__(self):
        return f'SpecAugmented()'


class YesNo(Dataset):
    """YesNo is a 60 example test dataset. Targets have been converted to
    backwards hebrew text, rather than the original ordinals.
    """

    def __init__(self, cfg):
        self.data = ta.datasets.YESNO(root=cfg.datasets_dir, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, _, y = self.data[idx]
        return torch.squeeze(x, 0), self.decode(y)

    def decode(self, y):
        return ' '.join(['כן' if el == 1 else 'לא' for el in y])[::-1]

    def __repr__(self):
        return f'YesNo()'
