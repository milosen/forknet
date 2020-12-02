import os
from typing import Union, Any

import torch
from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat


class NAMIC(Dataset):
    """NAMIC dataset."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform
        self.cases = ['case01015']
        self.case = 0
        self.n_slices = 176
        self.img_dims = (256, 256)
        self.mat_keys = dict(t1w='ana01', segGM='GM', segWM='WM')
        self.subject = dict(
            segWM=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}_WM')),
            segGM=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}_GM')),
            t1w=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}'))
        )

    def __len__(self):
        return len(self.cases)*self.n_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.case01015(idx)

    def case01015(self, idx):
        self.case = int(idx/self.n_slices)
        slice_in_case = idx % self.n_slices

        sample = {
            mod: {
                'data': self.subject[mod][
                            self.mat_keys[mod]
                        ][:, :, slice_in_case].reshape(
                    (1, self.img_dims[0], self.img_dims[1])
                ),
                'header': self.subject[mod]['__header__']
            } for mod in self.subject
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device="cuda"):
        self.device = torch.device(device)

    def __call__(self, sample):
        for key in sample:
            sample[key].update({
                'data': torch.from_numpy(sample[key]['data'].astype(np.float32)).to(self.device)
            })
        return sample


class Interpolate:
    """Interpolate."""
    def __init__(self, seg_factor: Union[float, tuple],
                 t1w_factor: Union[float, tuple]) -> None:
        self.seg_factor = seg_factor
        self.t1w_factor = t1w_factor

    def __call__(self, sample):
        for key in sample:
            if key == 't1w':
                factor = self.t1w_factor
            else:
                factor = self.seg_factor
            sample['seg'].update({
                'data': ndimage.zoom(sample[key]['data'], zoom=factor, order=0, mode='nearest')
            })
        return sample
