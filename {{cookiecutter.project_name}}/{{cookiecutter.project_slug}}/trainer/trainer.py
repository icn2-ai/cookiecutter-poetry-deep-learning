from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from {{cookiecutter.project_slug}}.base.base_trainer import BaseTrainer
from {{cookiecutter.project_slug}}.parse_config import ConfigParser
from {{cookiecutter.project_slug}}.utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metric_ftns: list[Callable],
        optimizer: torch.optim.Optimizer,
        config: ConfigParser,
        device: torch.device,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader | None = None,
        lr_scheduler: _LRScheduler | None = None,
        len_epoch: int | None = None,
        log_step: int | None = None,
    ) -> None:
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.val_data_loader = val_data_loader
        self.lr_scheduler = lr_scheduler
        if log_step is None:
            self.log_step = (
                int(np.sqrt(train_data_loader.batch_size)) if train_data_loader.batch_size is not None else 100
            )
        else:
            self.log_step = log_step

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # If it is a log_step and it is not the last batch, log the progress
            if batch_idx % self.log_step == 0 and batch_idx != self.len_epoch:
                self.logger.debug(f"Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}")
                log = self.train_metrics.result()
                wandb.log(log, step=(epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        val_log = self._valid_epoch(epoch)
        if self.val_data_loader is not None:
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            wandb.log(log, step=epoch * self.len_epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.val_data_loader is None:
            return {}
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for data, target in self.val_data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx: int) -> str:
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_data_loader, "n_samples"):
            current = (
                batch_idx * self.train_data_loader.batch_size if self.train_data_loader.batch_size is not None else 1
            )
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
