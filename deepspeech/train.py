"""
Training loop
"""

from collections import defaultdict
import math
import os

from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.cuda.amp as amp
from torch.utils.data.dataloader import DataLoader

from deepspeech import utils, datasets, decode


class Trainer:
    """Train deepspeech with mixed precision on a one cycle schedule.
    """

    def __init__(self, model, trainset, testset, cfg):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        self.model_cfg = model.cfg
        self.collate_fn = datasets.batch(model.cfg)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def checkpoint(self, name):
        base = wandb.run.dir if wandb.run.dir != '/' else '.'
        filename = os.path.join(base, self.cfg.ckpt_path(name))
        torch.save(self._model().state_dict(), filename)

    def _model(self):
        is_data_paralell = hasattr(self.model, 'module')
        return self.model.module if is_data_paralell else self.model

    def train(self):
        model, cfg, model_cfg = self.model, self.cfg, self.model_cfg
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas
        )

        # half precision gradient scaler
        scaler = amp.GradScaler(enabled=model_cfg.mixed_precision)

        # telemetry
        wandb.init(project=cfg.project_name)
        wandb.config.update(utils.cfgdict(model_cfg, cfg))
        wandb.config.update({'dataset': repr(self.trainset)})
        wandb.watch(model, log='all')
        wandb.save('checkpoints.*')

        # lr schedule
        schedule = utils.onecycle(optimizer, len(self.trainset), cfg)
        if cfg.finder:
            schedule = utils.lrfinder(optimizer, len(self.trainset), cfg)
            wandb.config.update({'dataset': 'lrfinder'})

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.trainset if is_train else self.testset
            metrics = utils.Metrics()
            loader = DataLoader(
                data,
                shuffle=is_train,
                pin_memory=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                collate_fn=self.collate_fn
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )

            for it, (x, y, nx, ny) in pbar:

                # placement
                x = x.to(self.device)
                y = y.to(self.device)
                nx = nx.to(self.device)
                ny = ny.to(self.device)

                with torch.set_grad_enabled(is_train):
                    with amp.autocast(enabled=model_cfg.mixed_precision):
                        logits, loss = model(x, y, nx, ny)

                    loss = loss.mean()  # collect gpus
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    if cfg.grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_norm_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    schedule.step()

                    # train logging
                    lr = schedule.get_last_lr()[0]
                    msg = f'{epoch+1}:{it} loss {loss.item():.5f} lr {lr:e}'
                    pbar.set_description(msg)
                    wandb.log({'learning rate': lr})
                    wandb.log({'train loss': loss})

                else:

                    # accumulate test metrics
                    x = decode.decode_argmax(model_cfg, x)
                    y = utils.decode_texts(model_cfg, y)
                    metrics.accumulate(x, y)

            return float(np.mean(losses))

        best = defaultdict(lambda: float('inf'))
        for epoch in range(cfg.max_epochs):

            train_loss = run_epoch('train')
            if train_loss < best['train']:
                best['train'] = train_loss
                self.checkpoint('best.train')

            if self.testset is not None:
                test_loss = run_epoch('test')
                wandb.log({'test loss': test_loss})
                wandb.log({'test metrics': metrics.to_dict()})
                if test_loss < best['test']:
                    best['test'] = test_loss
                    self.checkpoint('best.test')


class HParams(utils.HParams):

    # wandb project
    project_name = 'feldberlin-deepspeech'

    # once over the whole dataset, how many times max
    max_epochs = 10

    # number of examples in a single batch
    batch_size = 128

    # the learning rate
    learning_rate = 3e-4

    # adam betas
    betas = (0.9, 0.95)

    # training loop clips gradients
    grad_norm_clip = None

    # how many data loader threads to use
    num_workers = 0

    # is this a learning rate finder run
    finder = False

    # dataset splits
    splits = [0.8, 0.2]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def ckpt_path(self, name):
        return f'checkpoints.{name}'

    def n_steps(self, n_examples):
        return math.ceil(n_examples * self.max_epochs / self.batch_size)
