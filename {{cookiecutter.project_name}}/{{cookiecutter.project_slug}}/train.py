import argparse
import collections

import torch

import {{cookiecutter.project_slug}}.data.data_loaders as module_data
from {{cookiecutter.project_slug}}.model import loss as module_loss
from {{cookiecutter.project_slug}}.model import metric as module_metric
from {{cookiecutter.project_slug}}.model import model as module_arch
from {{cookiecutter.project_slug}}.parse_config import ConfigParser
from {{cookiecutter.project_slug}}.trainer import Trainer
from {{cookiecutter.project_slug}}.utils import prepare_device, seed


def main(config: ConfigParser) -> None:
    if config.config["seed"] is not None:
        seed(config.config["seed"])

    logger = config.get_logger("train")

    # setup data instances
    train_data_loader = config.init_obj("train_data_loader", module_data)
    val_data_loader = (
        config.init_obj("validation_data_loader", module_data) if config["validation_data_loader"] else None
    )

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_config = config["loss"]
    criterion = getattr(module_loss, loss_config["type"])(**loss_config["args"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="{{cookiecutter.project_name}} Train")
    args.add_argument(
        "-c", "--config", default="../configs/config.yaml", type=str, help="config file path (default: None)"
    )
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data;args;batch_size"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
