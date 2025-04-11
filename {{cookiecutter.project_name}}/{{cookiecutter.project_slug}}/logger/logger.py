import logging
import logging.config
from pathlib import Path
from typing import Union

import oyaml as yaml  # type: ignore[import-untyped]


def setup_logging(
    save_dir: Path, log_config: Union[str, Path] = "logger/logger_config.yaml", default_level: int = logging.INFO
) -> None:
    """
    Setup logging configuration
    """
    if isinstance(log_config, str):
        log_config = Path(log_config)

    if log_config.is_file():
        with open(log_config) as f:
            config = yaml.safe_load(f.read())
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}.")
        logging.basicConfig(level=default_level)
