"""
DeepSpeech https://arxiv.org/abs/1412.5567
"""

from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.cuda.amp as amp
from torchaudio.transforms import MelSpectrogram

from deepspeech import utils


class DeepSpeech(nn.Module):
    """DeepSpeech with small modifications.
    """

    def __init__(self, cfg, run_path=None):
        super().__init__()
        self.cfg = cfg

        # windowed inputs implemented as convolution
        self.conv = nn.Conv2d(1, cfg.n_hidden,
                              stride=cfg.stride,
                              kernel_size=(cfg.n_mels, cfg.kernel_width))

        self.dense_a = nn.Linear(cfg.n_hidden, cfg.n_hidden)
        self.dense_b = nn.Linear(cfg.n_hidden, cfg.n_hidden)

        # vanilla rnn as in paper
        self.gru = nn.RNN(cfg.n_hidden, cfg.n_hidden,
                          bidirectional=True, batch_first=True)

        # head
        self.dense_end = nn.Linear(cfg.n_hidden, cfg.n_graphemes())

        # load a checkpoint
        if run_path:
            utils.load_chkpt(self, run_path)

    def forward(self, x, nx, y=None, ny=None):
        "(B, H, W) batches of mfccs, (B, W) batches of graphemes."

        with amp.autocast(enabled=self.cfg.mixed_precision):
            B, H, W = x.shape

            # first convolution, collapses H
            x = F.pad(x, (self.cfg.padding(), self.cfg.padding(), 0, 0))
            x = F.relu(self.conv(x.unsqueeze(1)))  # add empty channel dim
            x = torch.squeeze(x, 2).permute(0, 2, 1)  # B, W, C

            # dense, gru
            x = F.relu(self.dense_a(x))
            x = F.relu(self.dense_b(x))
            x = F.relu(self.gru(x)[0])

            # sum over last dimension, fwd and bwd
            x = torch.split(x, self.cfg.n_hidden, dim=-1)
            x = torch.sum(torch.stack(x, dim=-1), dim=-1)

            # head
            x = F.relu(self.dense_end(x))
            x = F.log_softmax(x, dim=2)

            # loss
            loss = None
            if y is not None and ny is not None:
                nx = self.cfg.frame_lengths(nx)
                xctc = x.permute(1, 0, 2)  # W, B, C
                loss = F.ctc_loss(xctc, y, nx, ny)

            return x, loss


class HParams(utils.HParams):

    # use mixed precision
    mixed_precision = True

    # audio sampling rate
    sampling_rate = 8000

    # size of fft window in ms
    n_fft_ms = 20

    # number of mel filter banks
    n_mels = 20

    # convolution kernel width
    kernel_width = 9

    # network depth
    n_hidden = 2046

    # first layer stride. value of 2 will half the frequency
    stride = 2

    # graphemes. first char is blank
    graphemes = np.array([c for c in 'εabcdefghijklmnopqrstuvwxyz -'])

    # root directory to persist datsets in
    datasets_dir = '/tmp'

    # max specaugment frequency mask height, in number number of bins
    max_fq_mask = 10

    # max specaugment time mask width, in number number of frames
    max_time_mask = 10

    # dataset splits
    splits = [0.8, 0.2]

    # random seed
    seed = 2687

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def padding(self):
        return self.kernel_width // 2

    @lru_cache()
    def graphemes_idx(self):
        return {x:i for i,x in enumerate(self.graphemes)}

    def n_graphemes(self):
        return len(self.graphemes)

    def n_fft(self):
        return int(math.ceil(self.n_fft_ms * self.sampling_rate / 1000))

    def n_downsampled_frames(self, n_in):
        return int(math.ceil(n_in / self.stride))

    def frame_lengths(self, in_lengths):
        lengths = [self.n_downsampled_frames(x) for x in in_lengths]
        return torch.tensor(lengths)

    def mfcc_config(self):
        return {
            'sample_frequency': self.sampling_rate,
            'num_mel_bins': self.n_mels,
            'num_ceps': self.n_mels
        }
