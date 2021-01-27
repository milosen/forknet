import torch
import torch.utils.data
import numpy as np
from torchvision.transforms import Normalize


class ToTensor:
    """Convert all ndarrays in a sample to torch Tensors."""
    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample


class Norm:
    """Normalize input tensor."""
    def __init__(self, mean, std):
        self.normalize = Normalize(mean, std)

    def __call__(self, sample):
        sample['t1w'] = self.normalize(sample['t1w'])
        return sample
