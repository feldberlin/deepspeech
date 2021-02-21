import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn as nn
import torch.utils.data.dataset as td
import torchaudio as ta

from deepspeech import utils

# alphabet for the yesno dataset
YESNO_GRAPHEMES = np.array([c for c in 'εeklnor '])


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
        y = pad_sequence(y, batch_first=True, padding_value=0)

        return x, nx, y, ny

    return fn


# transforms and augmentations

def transform(cfg):
    return KaldiMFCC(cfg)


def spec_augment(cfg):
    return nn.Sequential(
        ta.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        ta.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
    )


class KaldiMFCC(nn.Module):
    "Use kaldi compatible mfcss, since they work better."

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = x.unsqueeze(0)  # introduce stereo dimension
        return ta.compliance.kaldi.mfcc(x, **self.cfg.mfcc_config()).T


# datasets

def splits(dataset, cfg):
    """Split according to cfg.splits. Uses cfg.seed for reproducible splits.
    """

    assert sum(cfg.splits) == 1.0
    gen = torch.Generator().manual_seed(cfg.seed)
    counts = [round(x * len(dataset)) for x in cfg.splits]
    return td.random_split(dataset, counts, generator=gen)


class SpecAugmented(Dataset):
    """SpecAugmentation as per https://arxiv.org/abs/1904.08779. Warping has
    not been implemented because it has a negligible impact and is not
    implemented in torchaudio.
    """

    def __init__(self, dataset, cfg, masked):
        super().__init__()
        self.spec_augment = spec_augment(cfg)
        self.dataset = dataset
        self.masked = masked

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        y = self.dataset[idx][1]
        if self.masked: x = self.spec_augment(x)
        return x, y


class YesNo(Dataset):
    """YesNo is a 60 example test dataset. Targets have been converted to
    english graphemes to avoid mysterious issues with right to left text.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ds = ta.datasets.YESNO(root=cfg.datasets_dir, download=True)
        self.transform = transform(cfg)
        self.sr = cfg.sampling_rate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        xin, sr, yin = self.ds[idx]
        assert sr == self.sr, sr
        x = self.transform(torch.squeeze(xin, 0))
        y = self.decode(yin)
        return x, y

    def decode(self, y):
        return ' '.join(['ken' if el == 1 else 'lor' for el in y])


class LibriSpeech(Dataset):
    """Librispeech english
    """

    def __init__(self, cfg):
        super().__init__()
        self.ds = ta.datasets.LIBRISPEECH(root=cfg.datasets_dir, download=True)
        self.transform = transform(cfg)
        self.sr = cfg.sampling_rate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, sr, y, speaker_id, chapter_id, utterance_id = self.ds[idx]
        assert sr == self.sr, sr
        x = self.transform(torch.squeeze(x, 0))
        y = y.lower().replace("'", '')
        return x, y
