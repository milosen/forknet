import matplotlib.pyplot as plt
from numpy import ndarray


def plot_matrix(mat: ndarray, title: str = '') -> None:
    """Plot a 2D numpy array like an image."""
    plt.figure().set_tight_layout(False)
    cax = plt.matshow(mat)
    plt.colorbar(cax)
    plt.title(title)
    plt.show()


def overlay(im1: ndarray, im2: ndarray, title: str = '') -> None:
    """Plot a 2D numpy array like an image."""
    plt.figure().set_tight_layout(False)
    cax = plt.imshow(im1, interpolation=None)
    plt.imshow(im2, interpolation=None, alpha=0.5)
    plt.colorbar(cax)
    plt.title(title)
    plt.show()
