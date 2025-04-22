# Testing {{cookiecutter.project_name}} 

## Overview

This file explains how to evaluate the trained model using the provided
testing script and configuration file.

---

## Configuration setup

The .yaml configuration files for testing can be the same as the ones used for
training. However, there
are a few things to consider in the testing configuration:

- The `arch` field must be the same as the one used for training. The model must be
  initialized with the same parameters.
- The `train_data_loader` and `validation_data_loader` fields are not necessary. A new
  field, `test_data_loader`, must be added with the same structure as the other two.
- The `optimizer`, `lr_scheduler` and `trainer` fields are not necessary.
- The `loss` and `metrics` fields will be used to calculate the loss and metrics for the
  test set, so keep them.

## Running the testing script

To run the testing script, execute the following command:

```bash
python test.py --config path/to/config.yaml --resume path/to/checkpoint.pth
```

The `--config` flag is used to specify the path to the configuration file. The `--resume`
flag is used to specify the path to the checkpoint file. After a training, the specific configuration file
is also stored together with the checkpoint files, with the name `config.yaml`. So you could use that one (adding the
`test_data_loader` field, if necessary) instead of looking for the original configuration file.

The testing script will load the model and the data loaders, and will evaluate the model on the test set. The results
will be printed on the console. For now, they are not saved to a file, but you can redirect the output to a file if you
want to save them.
