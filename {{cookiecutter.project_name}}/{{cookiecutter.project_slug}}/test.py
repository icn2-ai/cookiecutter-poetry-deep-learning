import argparse

import torch
from tqdm import tqdm

import {{cookiecutter.project_slug}}.data.data_loaders as module_data
import {{cookiecutter.project_slug}}.model.loss as module_loss
import {{cookiecutter.project_slug}}.model.metric as module_metric
import {{cookiecutter.project_slug}}.model.model as module_arch
from {{cookiecutter.project_slug}}.parse_config import ConfigParser


def main(config: ConfigParser) -> None:
    logger = config.get_logger("test")

    # setup data instances
    data_loader = config.init_obj("test_data_loader", module_data)

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_config = config["loss"]
    criterion = getattr(module_loss, loss_config["type"])(**loss_config["args"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info(f"Loading checkpoint: {config.resume} ...")
    if config.resume is None:
        message = "No checkpoint provided for testing."
        raise ValueError(message)
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="saved/models/CookieCutterDL/0411_123152/config.yaml",
        type=str,
        help="config file path, should include test data loader field (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="saved/models/CookieCutterDL/0411_123152/model_best.pth",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)
    main(config)

