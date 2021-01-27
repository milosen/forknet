import logging
import click
from forknet.train import train_net, print_network_and_exit
from forknet.tune import tune_net
from forknet.eval import test_net


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@cli.command(help="Tune the hyper-parameters", context_settings={'show_default': True})
@click.option('--max_epochs', default=200, help='number of max training epochs')
@click.option('--num_samples', default=2, help='number of samples from the hyper-parameter distribution')
@click.option('--gpus_per_trial', default=1, help='number of gpus per trial')
@click.option('--cpus_per_trial', default=1, help='number of cpus per trial')
@click.option('--distribute/--no-distribute', default=False, help='running ray tune in distributed mode')
def tune(max_epochs, num_samples, gpus_per_trial, cpus_per_trial, distribute):
    tune_net(max_epochs, num_samples, gpus_per_trial, cpus_per_trial, distribute)


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
def train(batch_size, epochs, lr, betas, eps, l2_penalty, amsgrad, load, checkpoint):
    train_net(batch_size, epochs, lr, betas, eps, l2_penalty, amsgrad, load, checkpoint)


@cli.command(help="Allocate network and print network information")
@click.option('--load', default=None, type=str,
              help='path to state dict file')
def dry_run(load):
    print_network_and_exit(load)


@cli.command(help="Test a trained network on the test data.")
@click.argument('load', type=str)
@click.option('--xslice', default=20, help="choose axial slice")
@click.option('--show/--no-show', default=30, help="show plots")
def test(load, xslice, show):
    test_net(load, xslice, show)
