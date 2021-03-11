from typing import List

import matplotlib.pyplot as plt
from numpy import ndarray
import torch

from forknet.datasets.miccai18 import MICCAI18


def plot_matrix(mat: ndarray, title: str = '', cmap: str = 'gray') -> None:
    """Plot a 2D numpy array like an image."""
    cax = plt.matshow(mat, cmap=cmap)
    plt.colorbar(cax)
    plt.title(title)
    plt.show()


def overlay(im1: ndarray, im2: ndarray, title: str = '') -> None:
    """Plot a 2D numpy array like an image."""
    cax = plt.imshow(im1, interpolation=None, cmap='gray')
    plt.imshow(im2, interpolation=None, alpha=0.5, cmap='GnBu')
    plt.colorbar(cax)
    plt.title(title)
    plt.show()


def softmax_rule(out: torch.Tensor, dim=0) -> List[torch.Tensor]:
    tensors = []
    for i in range(len(MICCAI18.tissues)):
        tensors.append((i == torch.argmax(out, dim=dim)).float().unsqueeze(dim=dim))
    return tensors
