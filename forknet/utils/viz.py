import matplotlib.pyplot as plt

from forknet.train import init_forknet
from torchvision.utils import make_grid
import logging
import sys
import contextlib
import os

import torch
import torch.utils.data
import torch.utils.tensorboard
from forknet.datasets.miccai18 import MICCAI18
import torch


def show_features(load, encoder=0, xslice=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_forknet(load, device)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # encode[0] is the convolution module of the encoder
    model.encoders[encoder].encode[0].register_forward_hook(get_activation('enc1'))
    dataset = MICCAI18(base_dir="./data/miccai18/training", case_list=['4'], slice_axis=2)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=48,
        shuffle=True
    )
    batch_dict = iter(loader).next()
    for tensor in batch_dict:
        batch_dict[tensor] = batch_dict[tensor].to(device)
    data = batch_dict['t1w'][xslice]
    data.unsqueeze_(0)
    output = model(data)

    act = activation['enc1'].squeeze()
    num_plots=2
    fig, axarr = plt.subplots(ncols=num_plots)
    for idx in range(num_plots):
        axarr[idx].imshow(act[idx].detach().cpu())
    plt.show()