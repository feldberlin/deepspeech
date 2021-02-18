import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn as nn
import torch.utils.data.dataset as td
import torchaudio as ta

from deepspeech import utils


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


# transforms

def spec_augment(cfg, masked=True):
    tt = nn.Sequential(
        ta.transforms.MFCC(sample_rate=cfg.sampling_rate, n_mfcc=20),
        ta.transforms.FrequencyMasking(freq_mask_param=cfg.max_fq_mask),
        ta.transforms.TimeMasking(time_mask_param=cfg.max_time_mask)
    )

    if masked:
        return tt
    else:
        return tt[:-2]


class Rescaled(nn.Module):
    "Log10 on the melspectogram, followed by scaling to roughly -1 to 1"

    def __init__(self):
        super().__init__()
        self.amp_to_db = ta.transforms.AmplitudeToDB()

    def forward(self, x):
        return self.amp_to_db(x) / 100.0  # DB scale is around -100 to 100


# datasets

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


class SpecAugmented(Dataset):
    """SpecAugmentation as per https://arxiv.org/abs/1904.08779. Warping has
    not been implemented because it has a negligible impact and is not
    implemented in torchaudio.
    """

    def __init__(self, dataset, cfg, masked):
        super().__init__()
        self.spec_augment = spec_augment(cfg, masked)
        self.dataset = dataset
        self.masked = masked

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        y = self.dataset[idx][1]
        return self.spec_augment(x), y

    def __repr__(self):
        return f'SpecAugmented()'


# alphabet for the yesno dataset
YESNO_GRAPHEMES = np.array([c for c in 'εeklnor '])


class YesNo(Dataset):
    """YesNo is a 60 example test dataset. Targets have been converted to
    english to avoid mysterious issues with right to left text.
    """


    def __init__(self, cfg):
        super().__init__()
        root=cfg.datasets_dir
        self.dataset = ta.datasets.YESNO(root=root, download=True)
        self.sr = cfg.sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, sr, y = self.dataset[idx]
        assert sr == self.sr, sr
        return torch.squeeze(x, 0), self.decode(y)

    def decode(self, y):
        return ' '.join(['ken' if el == 1 else 'lor' for el in y])

    def __repr__(self):
        return f'YesNo()'


class LibriSpeech(Dataset):
    """Librispeech english
    """

    def __init__(self, cfg):
        super().__init__()
        root=cfg.datasets_dir
        self.dataset = ta.datasets.LIBRISPEECH(root=root, download=True)
        self.sr = cfg.sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, sr, y, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        assert sr == self.sr, sr
        return torch.squeeze(x, 0), y.lower().replace("'", '')

    def __repr__(self):
        return f'LibriSpeech()'
