import torch
import torchaudio as ta
import torch.nn as nn
import torch.utils.data.dataset as td
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from deepspeech import utils

# datasets

def yesno(cfg):
    return YesNo(cfg)


def librispeech(cfg):
    root = cfg.datasets_dir
    return ta.datasets.LIBRISPEECH(root=root, download=True)


def commonvoice(cfg, lang='english'):
    root = cfg.datasets_dir
    return ta.datasets.COMMONVOICE(root=root, url=lang, download=True)


def splits(dataset, cfg, unmasked_trainset=False):
    """Split according to cfg.splits. Wraps each split in SpecAugment. The
    first split is the trainset and applies SpecAugment masking. Uses cfg.seed
    for reproducible splits.
    """

    assert sum(cfg.splits) == 1.0
    gen = torch.Generator().manual_seed(cfg.seed)
    counts = [round(x * len(dataset)) for x in cfg.splits]
    return [
        SpecAugmented(s, cfg, masked=not unmasked_trainset and i == 0)
        for i, s
        in enumerate(td.random_split(dataset, counts, generator=gen))
    ]


# transforms

def spec_augment(cfg, masked=True):
    tt = nn.Sequential(
        ta.transforms.MelSpectrogram(**cfg.mel_config()),
        ta.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        ta.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
    )

    if masked:
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

        return x, nx, y, ny

    return fn


# datasets

class SpecAugmented(Dataset):
    """SpecAugmentation as per https://arxiv.org/abs/1904.08779. Warping has
    not been implemented because it has a negligible impact and is not
    implemented in torchaudio.
    """

    def __init__(self, data, cfg, masked):
        self.spec_augment = spec_augment(cfg, masked)
        self.data = data
        self.masked = masked

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
