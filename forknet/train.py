import logging
import sys
from functools import partial
import contextlib

import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from tqdm import tqdm
import click
import numpy as np
import os
import ray.tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from forknet.model import ForkNet
from utils.data import MICCAI18
from utils.visuals import plot_matrix


def init_forknet(load, n_classes, device):
    net = ForkNet(n_classes=n_classes)
    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Loaded model from {load}')
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    return net


def validate_net(net,
                 val_loader,
                 criterion,
                 tissues,
                 device,
                 xslice=20):
    with torch.no_grad():
        validation_batch_dict = iter(val_loader).next()
        for tensor in validation_batch_dict:
            validation_batch_dict[tensor] = validation_batch_dict[tensor].to(device)
        validation_input = validation_batch_dict.pop('t1w')
        net_output = tuple(torch.sigmoid(o) for o in net(validation_input))
        losses = [
            criterion(
                net_output[i_tissue],
                validation_batch_dict[tissue]
            ) for i_tissue, tissue in enumerate(tissues)
        ]
        input_sample = validation_input[xslice]
        sample_targets = tuple(validation_batch_dict[tissue][xslice] for tissue in tissues)
        sample_outputs = tuple(o[xslice] for o in net_output)
        # write network graph, visualization and metrics to tensorboard
        img_grid = [(input_sample - input_sample.min()) / torch.max(input_sample)]
        for i_tissue, _ in enumerate(tissues):
            img_grid.append(sample_targets[i_tissue])
        img_grid.append(torch.zeros((1, 256, 256), device=device))
        for i_tissue, tissue in enumerate(tissues):
            img_grid.append(sample_outputs[i_tissue])
        return losses, img_grid


def tensorboard_write(writer,
                      global_step,
                      tissues,
                      losses,
                      val_losses,
                      img_grid,
                      net,
                      device,
                      write_network_graph=False):
    for i_tissue, tissue in enumerate(tissues):
        writer.add_scalars(
            f'loss/{tissue}', {
                'train': losses[i_tissue],
                'validation': val_losses[i_tissue],
            },
            global_step=global_step
        )
    writer.add_image(
        img_tensor=torchvision.utils.make_grid(img_grid, nrow=len(tissues) + 1),
        tag=f'validation sample {tissues}',
        global_step=global_step
    )
    if (global_step % 100) == 0:
        gradients = torch.cat([param.grad.view(-1) for param in net.parameters()])
        writer.add_histogram(f'Gradients', gradients, global_step=global_step)
    if write_network_graph and global_step == 0:
        writer.add_graph(net, torch.zeros((1, 1, 256, 256), device=device))


def train_net(batch_size,
              epochs,
              lr,
              betas,
              eps,
              l2_penalty,
              amsgrad,
              load=None,
              checkpoint=None,
              use_raytune=False,
              data_dir='data/miccai18/training'):
    # test_case = ['1']
    validation_cases = ['4']
    train_cases = ['5', '7', '14', '070', '148']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MICCAI18(base_dir=data_dir, case_list=train_cases)
    val_dataset = MICCAI18(base_dir=data_dir, case_list=validation_cases)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset),
                                             shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    net = ForkNet(n_classes=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=l2_penalty, amsgrad=amsgrad)
    writer = torch.utils.tensorboard.SummaryWriter()
    if load:
        state = torch.load(load, map_location=device)
        if state is tuple:
            model_state, optimizer_state = state
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            logging.info(f'Loaded model and optimizer from {load}')
        else:
            net.load_state_dict(state)
            logging.info(f'Loaded model from {load}')
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    if not use_raytune:
        logging.info(f'Network:\n'
                     f'\t{net.n_classes} decoder tracks: {train_dataset.tissues}')
        logging.info(f'''Training:
                Epochs:             {epochs}
                Batch size:         {batch_size}
                Learning rate:      {lr}
                Betas:              {betas}
                Eps:                {eps}
                L2 Penalty:         {l2_penalty}
                Using AMSGrad:      {amsgrad}
                Device:             {device.type}
            ''')

    try:
        global_step = 0
        for epoch in range(epochs):
            epoch_bar = tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit=' slices')
            with contextlib.nullcontext if use_raytune else epoch_bar as bar:
                for batch_dict in train_loader:
                    for mod in batch_dict:
                        batch_dict[mod] = batch_dict[mod].to(device)
                    inp = batch_dict.pop('t1w')
                    out = net(inp)
                    losses = [
                        criterion(
                            out[i_tissue],
                            batch_dict[tissue]
                        ) for i_tissue, tissue in enumerate(train_dataset.tissues)
                    ]
                    optimizer.zero_grad()
                    torch.autograd.backward(losses)
                    optimizer.step()
                    bar.update(inp.shape[0]) if use_raytune else None
                    val_losses, img_grid = validate_net(
                        net=net, val_loader=val_loader, criterion=criterion,
                        device=device, tissues=train_dataset.tissues
                    )
                    if not use_raytune:
                        tensorboard_write(
                            writer=writer,
                            global_step=global_step,
                            tissues=train_dataset.tissues,
                            losses=losses, val_losses=val_losses,
                            img_grid=img_grid, net=net,
                            device=device, write_network_graph=True
                        )
                    global_step += 1
            if use_raytune:
                with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((net.state_dict(), optimizer.state_dict()), path)
                ray.tune.report(loss=sum([loss.item() for loss in val_losses]))
            else:
                if checkpoint and (epoch % checkpoint) == (checkpoint - 1):
                    torch.save(net.state_dict(), 'scheduled_checkpoint.pth')
                    logging.info(f'Saved checkpoint for epoch {epoch + 1}')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'interrupt.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
    finally:
        writer.flush()
        writer.close()


def tune_net(config, checkpoint_dir=None, data_dir=None):
    train_net(config['batch_size'], config['epochs'], config['lr'], [config['beta1'], config['beta2']], config['eps'], config['l2_penalty'],
              config['amsgrad'], use_raytune=True, load=checkpoint_dir, data_dir=data_dir)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@cli.command(help="Tune the hyperparameters")
@click.option('--max_epochs', default=1, help='number of max training epochs')
@click.option('--num_samples', default=2, help='number of samples from the hyper-parameter distribution')
@click.option('--gpus_per_trial', default=1, help='number of gpus per trial')
@click.option('--cpus_per_trial', default=1, help='number of gpus per trial')
def tune(max_epochs, num_samples, gpus_per_trial, cpus_per_trial):
    data_dir = os.path.abspath("./data/miccai18/training/")
    config = {
        "lr": ray.tune.loguniform(1e-4, 1e-1),
        "beta1": ray.tune.loguniform(1e-4, 1e-1),
        "beta2": ray.tune.loguniform(1e-4, 1e-1),
        "eps": ray.tune.loguniform(1e-4, 1e-1),
        "l2_penalty": ray.tune.loguniform(1e-4, 1e-1),
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
    result = ray.tune.run(
        partial(tune_net, data_dir=data_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = ForkNet(n_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and gpus_per_trial > 1:
        best_trained_model = torch.nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)


@cli.command(help="Train the network")
@click.option('-b', '--batch_size', default=20, help='number of slices in batch')
@click.option('--epochs', default=100, help='number of training epochs')
@click.option('-l', '--lr', default=1e-3, help='learning rate')
@click.option('--eps', default=1e-8, help='adam epsilon value')
@click.option('--betas', default=(0.9, 0.999), help='adam beta values')
@click.option('--l2_penalty', default=0., help='adam weight decay (L2 penalty)')
@click.option('--amsgrad/--no-amsgrad', default=False, help='use amsgrad version of adam from the '
                                                            'paper `On the Convergence of Adam and Beyond`')
@click.option('--load', default=None, type=str,
              help='path to state dict file')
@click.option('--checkpoint', default=None, type=int,
              help='save network state dict every N epochs')
def train(batch_size,
          epochs,
          lr,
          betas,
          eps,
          l2_penalty,
          amsgrad,
          load,
          checkpoint):
    train_net(batch_size, epochs, lr, betas, eps, l2_penalty, amsgrad, load, checkpoint)


@cli.command(help="Allocate network and print network information")
@click.option('--load', default=None, type=str,
              help='path to state dict file')
@click.option('--n_classes', default=2, type=int,
              help='path to state dict file')
@click.option('--force_cpu', default=False,
              help="always use cpu, even if cuda available")
def dry_run(load, n_classes, force_cpu):
    device = torch.device('cuda' if torch.cuda.is_available() and force_cpu is False else 'cpu')
    print(init_forknet(load, n_classes, device))


@cli.command(help="Test a trained network on the test data.")
@click.argument('load', type=str)
@click.option('--n_classes', default=2, type=int,
              help='path to state dict file')
@click.option('--force_cpu', default=False,
              help="always use cpu, even if cuda available")
@click.option('--xslice', default=30, help="choose axial slice")
def test_net(load, n_classes, force_cpu, xslice):
    test_case = ['1']
    device = torch.device('cuda' if torch.cuda.is_available() and force_cpu is False else 'cpu')
    dataset = MICCAI18(base_dir='data/miccai18/training', case_list=test_case)
    net = init_forknet(load=load, n_classes=n_classes, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    _, img_grid = validate_net(net=net, val_loader=test_loader, criterion=criterion,
                               tissues=dataset.tissues, device=device, xslice=xslice)
    plot_matrix(torchvision.utils.make_grid(img_grid, nrow=len(dataset.tissues) + 1).cpu().numpy()[0])
