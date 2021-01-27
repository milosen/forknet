import os
from functools import partial
import torch

import ray.tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from forknet.model import ForkNet
from forknet.train import train_net
from forknet.datasets.miccai18 import MICCAI18


def tune_func(config, checkpoint_dir=None, data_dir=None):
    train_net(config['batch_size'], config['epochs'], config['lr'],
              betas=[config['beta1'], config['beta2']], eps=config['eps'],
              l2_penalty=config['l2_penalty'], amsgrad=config['amsgrad'],
              use_raytune=True, load=checkpoint_dir, data_dir=data_dir)


def tune_net(max_epochs, num_samples, gpus_per_trial, cpus_per_trial, distribute):
    data_dir = os.path.abspath("./data/miccai18/training/")
    config = {
        "lr": ray.tune.loguniform(1e-4, 1e-1),
        "beta1": ray.tune.loguniform(1e-1, 1),
        "beta2": ray.tune.loguniform(1e-1, 1),
        "eps": ray.tune.choice([1e-8]),
        "l2_penalty": ray.tune.loguniform(1e-4, 1),
        "batch_size": ray.tune.choice([20, 30, 60]),
        "amsgrad": ray.tune.choice([True, False]),
        "epochs": ray.tune.choice([max_epochs])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"]
    )
    if distribute:
        ray.init(address='auto')
    result = ray.tune.run(
        partial(tune_func, data_dir=data_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./ray_results"
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = ForkNet(tissues=MICCAI18.tissues)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and gpus_per_trial > 1:
        best_trained_model = torch.nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
