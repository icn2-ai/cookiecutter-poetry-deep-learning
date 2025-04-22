# Training {{cookiecutter.project_name}}

## Overview

This file explains how to train the RADIAN model using the provided training script, configuration file, and experiment
tracking setup.

---

## Configuration setup

First, let's explore the .yaml configuration files for training. They are stored in
`configs/` and they have the following structure:

```yaml
name: CookieCutterDL
experiment_group: v0.0
n_gpu: 1

seed: null # Fix seed for reproducibility

arch:
  type: IdentityModel  # Name of the model, should be one of the architectures in model.py
  args: {}

train_data_loader:
  type: RandomVectorDataLoader  # Name of the data loader, should be one of the data loaders in data_loaders.py
  args:  # Arguments for the data loader
    num_samples: 1000
    vector_size: 100

validation_data_loader:
  type: RandomVectorDataLoader
  args:
    num_samples: 1000
    vector_size: 100

optimizer:
  type: Adam
  args:
    lr: 0.001
    weight_decay: 0
    amsgrad: true

loss:
  type: "MSELoss"  # Name of the loss function, should be one of the losses in loss.py
  args: {}

metrics:
  - mse_metric  # Name of the metric, should be one of the metrics in metric.py

lr_scheduler:
  type: StepLR
  args:
    step_size: 50
    gamma: 0.1

trainer:
  epochs: 5
  save_dir: "saved/"
  save_period: 1
  verbosity: 2
  monitor: "min val_loss"
  early_stop: 10
  wandb: False  # Enable WandB tracking
  codecarbon_save_api: false  # Enable CodeCarbon tracking

```

Let's look at the more important fields one by one:

- `arch`: This field specifies the model to be used. The `type` field should be one of the architectures specified in
  `model.py`, and the `args` field should be the arguments for the initialization of the model.
- `train_data_loader` and `validation_data_loader`: These fields specify the configuration for the data loaders. The
  `type` field should be one of the data loaders specified in `data_loaders.py`, and
  the `args` field should be the arguments for the initialization of the data loader. 
- `optimizer`: This field specifies the optimizer to be used. They are from the `torch.optim` package.
- `loss`: This field specifies the loss function to be used. The `type` field should be one of the losses specified in
  `loss.py`, and the `args` field should be the arguments for the initialization of the loss.
- `metrics`: This field specifies the metrics to be used. They should be specified as their names are in the file
  `metric.py`.
- `lr_scheduler`: This field specifies the learning rate scheduler to be used. They are from the `torch.optim.lr_scheduler`
- `trainer`: This field specifies the training configuration. The arguments are mostly self-explanatory, and the
  comments above should help understand them.

## Running the training script

To train the model, you can run the following command:

```bash
python {{cookiecutter.project_slug}}/train.py --config configs/config.yaml
```

This will start the training process. The training information will be logged to the console and to the directory
specified in the `save_dir` field of the configuration file. The model weights will be saved every `save_period` epochs,
and a sample from the validation set will be plotted every `save_period` epochs. Every `log_step` steps, the training
loss will be logged to the console. In the next sections, we will explain how to track the training process using
WandB and CodeCarbon.

---

## Experiment tracking with Weights&Biases

To track the training process, we use Weights&Biases. If everything has been setup correctly, 
the wandb package should already be installed. To start tracking the training,
you should have a WandB account. You can create one [here](https://wandb.ai/site). Once you have it, you can run the
following command to login:

```bash
wandb login
```

It will ask you for account name and your WandB API key, which you can find in your account settings. Once that is done,
you can go back to the .yaml file. The following fields are relevant for WandB tracking:

- `name`: This will be the name of the project in WandB.
- `experiment_group`: This will be the group of experiments the current experiment will belong to.
- `wandb` (inside the trainer configuration): This should be set to `true` to enable WandB tracking.

Once you have set these fields, you can run the training script as explained above. You should see the training process
in the WandB dashboard. The project name will be the one you specified in the `name` field, and you will be able
to see the training and validation losses, along with the specified metrics for each experiment.

---

## Emissions tracking with CodeCarbon

To track the carbon footprint of the training process, we use [CodeCarbon](https://codecarbon.io/). Again, if everything
has been setup correctly, the codecarbon package should
already be installed. Now, you need to configure it. To do so, follow the next steps:

1. Create an account on [codecarbon.io](https://codecarbon.io/).
2. Run the following command:

```bash
codecarbon login
```

This will redirect you to the codecarbon.io website to login. After that, a credentials.json file will be created in the
root of your project. You can test if the configuration was successful by running:

```bash
codecarbon test-api
```

The name, description and ID of your default organization should be printed in the terminal. 3. Ask the organization admin to add you to the organization you belong to. Once that is done, run again the test-api command to
check if you are in the organization. 4. Run the following command to set a .codecarbon.config file in the root of your project:

```bash
codecarbon config
```

When running the command, you will be asked to specify some information:

- Organization: pick an existing one (ICN2) or create a new one
- Project: pick an existing one (radian) or create a new one
- Experiment: pick an existing one (Training) or create a new one

5. Set the `codecarbon_save_api` field in the trainer configuration to `true`. If it is set to `false`, you will be able
   to see the carbon footprint of the training process in the console, but it will not be uploaded to the CodeCarbon dashboard.

Once you have completed these steps, you are ready to start tracking the carbon footprint of your project.
To get a better understanding of how to use codecarbon, check the [documentation](https://docs.codecarbon.io/).
For now, only the track_emissions decorator is being used, in the train function inside the BaseTrainer class.