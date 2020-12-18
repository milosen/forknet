from datetime import datetime
import os
import logging
import sys
from typing import Any

import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from tqdm import tqdm
import click

from forknet.model import ForkNet
from utils.data import MICCAI18


def init_forknet(load: Any, n_classes: int,
                 device: torch.device) -> torch.nn.Module:
    net = ForkNet(n_classes=n_classes)
    if load:
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
        logging.info(f'Model loaded from {load}')
    net.to(device)
    return net


def random_data_split(dataset, split, batch_size):
    train, val = torch.utils.data.random_split(dataset, split)
    val_loader = torch.utils.data.DataLoader(val, batch_size=len(val), shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    return val_loader, train_loader


def validate_net(net,
                 val_loader,
                 criterion,
                 tissues,
                 device):
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
        input_sample = validation_input[0]
        sample_targets = tuple(validation_batch_dict[tissue][0] for tissue in tissues)
        sample_outputs = tuple(o[0] for o in net_output)
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
        gradients = [param.grad.view(-1) for param in net.parameters()]
        gradients = torch.cat(gradients)
        writer.add_histogram(f'Gradients', gradients, global_step=global_step)
    if write_network_graph:
        writer.add_graph(net, torch.zeros((1, 1, 256, 256)))


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@cli.command(help="Train the network")
@click.option('-b', '--batch_size', default=20, help='number of slices in batch')
@click.option('--epochs', default=20, help='number of training epochs')
@click.option('-l', '--lr', default=1e-3, help='learning rate')
@click.option('--eps', default=1e-8, help='adam epsilon value')
@click.option('--betas', default=(0.9, 0.999), help='adam beta values')
@click.option('--l2_penalty', default=0., help='adam weight decay (L2 penalty)')
@click.option('--amsgrad/--no-amsgrad', default=False, help='use amsgrad version of adam from the '
                                                            'paper `On the Convergence of Adam and Beyond`')
@click.option('--n_train', default=335, help='number of training slices')
@click.option('--n_val', default=1, help='number of validation slices')
@click.option('--force_cpu', default=False,
              help="always use cpu, even if cuda available")
@click.option('--runs_dir', default='runs',
              help='path to tensorboard runs directory')
@click.option('--load', default=None, type=str,
              help='path to state dict file')
@click.option('--checkpoint', default=None, type=int,
              help='save network state dict every N epochs')
def train_net(batch_size,
              epochs,
              lr,
              betas,
              eps,
              l2_penalty,
              amsgrad,
              n_train, n_val,
              force_cpu,
              runs_dir,
              load,
              checkpoint):
    # test_case = ['1']
    train_cases = ['4', '5', '7', '14', '070', '148']
    device = torch.device('cuda' if torch.cuda.is_available() and force_cpu is False else 'cpu')
    dataset = MICCAI18(base_dir='data/miccai18/training', case_list=train_cases)

    try:
        assert n_train + n_val == len(dataset)
    except AssertionError:
        logging.info(f"Dataset size: {len(dataset)}")
        raise

    net = init_forknet(load=load, n_classes=2, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=l2_penalty,
                                 amsgrad=amsgrad)

    logging.info(f'Network:\n'
                 f'\t{net.n_classes} decoder tracks: {dataset.tissues}')
    logging.info(f'''Training:
            Epochs:             {epochs}
            Batch size:         {batch_size}
            Learning rate:      {lr}
            Betas:              {betas}
            Eps:                {eps}
            L2 Penalty:         {l2_penalty}
            Using AMSGrad:      {amsgrad}
            Training size:      {n_train}
            Validation size:    {n_val}
            Device:             {device.type}
            Tensorboard data:   {runs_dir}
        ''')

    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(runs_dir, datetime.now().strftime("%Y-%m-%d_%H-%M"))
    )

    val_loader, train_loader = random_data_split(
        dataset=dataset,
        split=[n_train, n_val],
        batch_size=batch_size
    )

    try:
        global_step = 0
        for epoch in range(epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit=' slices') as bar:
                for batch_dict in train_loader:
                    for mod in batch_dict:
                        batch_dict[mod] = batch_dict[mod].to(device)
                    inp = batch_dict.pop('t1w')
                    out = net(inp)
                    losses = [
                        criterion(
                            out[i_tissue],
                            batch_dict[tissue]
                        ) for i_tissue, tissue in enumerate(dataset.tissues)
                    ]
                    torch.autograd.backward(losses)
                    optimizer.step()
                    bar.update(inp.shape[0])
                    del inp, out
                    val_losses, img_grid = validate_net(
                        net=net, val_loader=val_loader, criterion=criterion,
                        device=device, tissues=dataset.tissues
                    )
                    tensorboard_write(
                        writer=writer,
                        global_step=global_step,
                        tissues=dataset.tissues,
                        losses=losses, val_losses=val_losses,
                        img_grid=img_grid, net=net, write_network_graph=False
                    )
                    global_step += 1
            if checkpoint and (epoch % checkpoint) == (checkpoint - 1):
                torch.save(net.state_dict(), 'checkpoint.pth')
                logging.info(f'Saved checkpoint for epoch {epoch + 1}')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'interrupt.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
    finally:
        writer.flush()
        writer.close()


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
