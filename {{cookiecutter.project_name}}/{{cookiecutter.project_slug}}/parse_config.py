import logging
import os
import shutil
from datetime import datetime
from functools import partial, reduce
from logging import Logger
from operator import getitem
from pathlib import Path
from typing import Any

import oyaml as yaml  # type: ignore[import-untyped]

from {{cookiecutter.project_slug}}.logger import setup_logging


class ConfigParser:
    def __init__(
        self,
        config: dict[Any, Any],
        resume: Path | None = None,
        modification: dict[Any, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.yaml` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / "models" / exper_name / run_id
        self._log_dir = save_dir / "log" / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args: Any, options: Any = "") -> Any:
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.yaml"
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c ../config/config.yaml', for example."
            if args.config is None:
                raise ValueError(msg_no_cfg)

            resume = None
            cfg_fname = Path(args.config)

        with open(cfg_fname) as f:
            config = yaml.safe_load(f)

        if args.config and resume:
            # update new config for fine-tuning
            with open(args.config) as f:
                config.update(yaml.safe_load(f))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name: str, module: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        if not all(k not in module_args for k in kwargs):
            error_msg = "Overwriting kwargs given in config file is not allowed"
            raise ValueError(error_msg)
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name: str, module: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        if not all(k not in module_args for k in kwargs):
            error_msg = "Overwriting kwargs given in config file is not allowed"
            raise ValueError(error_msg)
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str) -> Any:
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name: str, verbosity: int = 2) -> Logger:
        msg_verbosity = f"verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}."
        if verbosity not in self.log_levels:
            raise ValueError(msg_verbosity)
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def update_dir(self, name: str) -> None:
        """
        Update the last directory of both save_dir and log_dir with the name given.
        """
        # Delete the created directories that were created with os.
        shutil.rmtree(self.save_dir)
        shutil.rmtree(self.log_dir)

        # Update the last directory of both save_dir and log_dir with the name given
        self._save_dir = self.save_dir.parent / name
        self._log_dir = self.log_dir.parent / name

        # Make the new directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # setting read-only attributes
    @property
    def config(self) -> dict:
        return self._config

    @property
    def save_dir(self) -> Any:
        return self._save_dir

    @property
    def log_dir(self) -> Any:
        return self._log_dir

    @property
    def plot(self) -> Any:
        return self.config["trainer"]["plot"]


# helper functions to update config dict with custom cli options
def _update_config(config: dict, modification: dict | None) -> dict:
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags: str) -> str:
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree: Any, keys: Any, value: Any) -> None:
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree: Any, keys: Any) -> Any:
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
