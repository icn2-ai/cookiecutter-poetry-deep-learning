from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import wandb
from codecarbon import track_emissions  # type: ignore[import-untyped]
from numpy import inf

from {{cookiecutter.project_slug}}.parse_config import ConfigParser
from {{cookiecutter.project_slug}}.utils import MetricTracker


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metric_ftns: list[Callable],
        optimizer: torch.optim.Optimizer,
        config: ConfigParser,
    ) -> None:
        """
        Base class for all trainers
        """
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0.0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            if self.mnt_mode not in ["min", "max"]:
                raise ValueError(f"{self.mnt_mode}")

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        self.wandb = cfg_trainer["wandb"]
        self.wandb_init = False

        self.codecarbon_save_api = cfg_trainer.get("codecarbon_save_api", False)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _create_metric_trackers(self) -> None:
        """
        Create MetricTracker instances for training and validation
        """
        self.train_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns])

    @staticmethod
    def dynamic_track_emissions(func: Callable) -> Callable:
        """
        Decorator to dynamically track emissions during training. This is useful when you want to enable/disable
        tracking emissions to the CodeCarbon API during training.
        """

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            decorated_func = track_emissions(save_to_api=self.codecarbon_save_api)(func)
            return decorated_func(self, *args, **kwargs)

        return wrapper

    def _setup_wandb(self) -> None:
        """
        Setup wandb for logging metrics. This function initializes wandb and sets the project name and group.
        """
        if self.wandb and not self.wandb_init:
            run = wandb.init(
                project=self.config.config["name"], group=self.config["experiment_group"], config=self.config.config
            )
            if run:  # Check if wandb is running
                self.logger.info(f"Wandb initialized with run id: {run.id}")
                self.wandb_init = True
                self.config.update_dir(run.name)
                self.checkpoint_dir = self.config.save_dir
        elif not self.wandb:
            self.logger.warning("Wandb is not enabled. Enable it in config file to log metrics to wandb.")
            wandb.init(mode="disabled")

    @dynamic_track_emissions
    def train(self) -> None:
        """
        Full training logic
        """
        not_improved_count = 0

        self._setup_wandb()

        # Create MetricTracker instances for training and validation
        self._create_metric_trackers()

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f"    {key!s:15s}: {value}")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        f"Warning: Metric '{self.mnt_metric}' is not found. "
                        "Model performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        f"Validation performance didn't improve for {self.early_stop} epochs. " "Training stops."
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path: Path) -> None:
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
