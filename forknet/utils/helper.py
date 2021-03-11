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
    plt.imshow(im2, interpolation=None, alpha=0.7, cmap='GnBu')
    plt.colorbar(cax)
    plt.title(title)
    plt.show()


def softmax_rule(out: torch.Tensor, dim=0) -> List[torch.Tensor]:
    tensors = []
    for i in range(len(MICCAI18.tissues)):
        tensors.append((i == torch.argmax(out, dim=dim)).float().unsqueeze(dim=dim))
    return tensors


def load_volume(file: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(file, map_location=device).squeeze().detach().cpu()


def show(img: ndarray):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()


def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols))
    return result


def dice_coefficient(pred, gt, smooth=1e-5):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    n = gt.size(0)
    pred_flat = pred.reshape(n, -1)
    gt_flat = gt.reshape(n, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / n
