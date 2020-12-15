from datetime import datetime
import os
import logging
import sys

import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from tqdm import tqdm
import click

from forknet.model import ForkNet
from utils.data import MICCAI18


def init_forknet(load: bool, n_classes: int,
                 device: torch.device) -> torch.nn.Module:
    net = ForkNet(n_classes=n_classes)
    if load:
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
        logging.info(f'Model loaded from {load}')
    net.to(device)
    return net


def split_data(dataset, split, batch_size):
    train, val = torch.utils.data.random_split(dataset, split)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    return val_loader, train_loader


def validate_net(net,
                 val_loader,
                 criterion,
                 writer,
                 tissues,
                 global_step,
                 write_graph,
                 device):
    with torch.no_grad():
        sample_val = iter(val_loader).next()
        for mod in sample_val:
            sample_val[mod] = sample_val[mod].to(device)
        inp_val = sample_val.pop('t1w')
        out = tuple(torch.sigmoid(o) for o in net(inp_val))
        losses = [
            criterion(
                out[i_tissue].squeeze(dim=1),
                sample_val[tissue].squeeze(dim=1).to(device)
            ) for i_tissue, tissue in enumerate(tissues)
        ]
        # write network graph, visualization and metrics to tensorboard
        img_grid = [inp_val[0]/torch.max(inp_val[0])]
        for i_tissue, tissue in enumerate(tissues):
            img_grid.append(sample_val[tissue][0])
        img_grid.append(torch.zeros((1, 256, 256), device=device))
        for i_tissue, tissue in enumerate(tissues):
            img_grid.append(out[i_tissue][0])
            writer.add_scalar(f'loss_{tissue}', losses[i_tissue], global_step=global_step)
        writer.add_image(
            img_tensor=torchvision.utils.make_grid(img_grid, nrow=len(tissues)+1),
            tag=f'data {tissues}',
            global_step=global_step
        )
        if write_graph:
            writer.add_graph(net, inp_val)


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@cli.command(help="Train the network")
@click.option('-b', '--batch_size', default=20, help='number of slices in batch')
@click.option('-epoch', '--epochs', default=20, help='number of training epochs')
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
              help='path to tensorboard runs directory')
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

    device = torch.device('cuda' if torch.cuda.is_available() and force_cpu is False else 'cpu')
    dataset = MICCAI18('data/miccai18/training')

    assert n_train + n_val == len(dataset)

    net = init_forknet(load=load, n_classes=2, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=l2_penalty,
                                 amsgrad=amsgrad)
    val_loader, train_loader = split_data(
        dataset=dataset, split=[n_train, n_val], batch_size=batch_size
    )

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

    try:
        for epoch in range(epochs):
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit=' slices') as bar:
                for batch_dict in train_loader:
                    for mod in batch_dict:
                        batch_dict[mod] = batch_dict[mod].to(device)
                    inp = batch_dict.pop('t1w')
                    out = net(inp)
                    losses = [
                        criterion(
                            out[i_tissue].squeeze(dim=1),
                            batch_dict[tissue].squeeze(dim=1)
                        ) for i_tissue, tissue in enumerate(dataset.tissues)
                    ]
                    torch.autograd.backward(losses)
                    optimizer.step()
                    bar.update(inp.shape[0])
                    del inp, out
                validate_net(net, val_loader, criterion, writer, tissues=dataset.tissues, global_step=epoch,
                             write_graph=(epoch == epochs - 1), device=device)
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


if __name__ == '__main__':
    train_net()
