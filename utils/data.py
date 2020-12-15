import os
from typing import Union, Any

import torch
from scipy import ndimage
import torch.utils.data
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from torchvision.transforms import Compose, Normalize


class NAMIC(torch.utils.data.Dataset):
    """NAMIC dataset."""
    def __init__(self, transform: Any) -> None:
        self.transform = transform
        self.cases = ['case01015']
        self.case = 0
        self.n_slices = 176
        self.img_dims = (256, 256)
        self.subject = dict(
            WM=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}_WM'))['WM'],
            GM=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}_GM'))['GM'],
            t1w=loadmat(os.path.join(self.cases[self.case], f'{self.cases[self.case]}'))['ana01']
        )
        # TODO better way?
        self.tissues = ['WM', 'GM']

    def __len__(self):
        return len(self.cases)*self.n_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.case01015(idx)

    def case01015(self, idx):
        # TODO load new case into memory when case number changes? Better way of doing this?
        self.case = int(idx/self.n_slices)
        slice_in_case = idx % self.n_slices

        sample = {
            mod: self.subject[mod][:, :, slice_in_case].transpose().reshape(
                    (1, self.img_dims[0], self.img_dims[1])
                ) for mod in self.subject
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MICCAI18(torch.utils.data.Dataset):
    """MICCAI18 dataset."""
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir
        self.cases = ['1', '4', '5', '7', '14', '070', '148']
        self.case = 0
        self.n_slices = 48
        self.img_dims = (256, 256)
        self.labels = dict(GM=1, WM=3)
        # we will need ordered tissue types
        self.tissues = [x[0] for x in sorted(self.labels.items())]
        # stack all the segmented axial slices from all subjects
        seg = np.concatenate(tuple(
            np.pad(
                np.array(nib.load(
                    os.path.join(self.base_dir, case, 'segm.nii.gz')
                ).dataobj), ((8, 8), (8, 8), (0, 0))
            ) for case in self.cases
        ), axis=2)
        self.data = dict(
            WM=np.zeros((self.img_dims[0], self.img_dims[1], self.n_slices * len(self.cases))),
            GM=np.zeros((self.img_dims[0], self.img_dims[1], self.n_slices * len(self.cases))),
            t1w=np.concatenate(tuple(
                np.pad(
                    np.array(nib.load(
                        os.path.join(self.base_dir, case, 'orig', 'reg_T1.nii.gz')
                    ).dataobj), ((8, 8), (8, 8), (0, 0))
                ) for case in self.cases
            ), axis=2)
        )
        # create binary label maps
        for tissue in self.tissues:
            label = self.labels[tissue]
            self.data[tissue][seg == label] = 1
        self.mean = np.mean(self.data['t1w'])
        self.std = np.std(self.data['t1w'])
        self.transform = Compose([
            ToTensor(torch.device('cpu')),
            Norm(self.mean, self.std)]
        )

    def __len__(self):
        return len(self.cases)*self.n_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.get_slice(idx)

    def get_slice(self, idx):
        sample = {
            mod: self.data[mod][:, :, idx].reshape(
                    # prepend sample dimension for batching
                    (1, self.img_dims[0], self.img_dims[1])
                ) for mod in self.data
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """Convert all ndarrays in a sample to torch Tensors."""
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.from_numpy(sample[key].astype(np.float32)).to(self.device)
        return sample


class Norm:
    """Convert all ndarrays in a sample to torch Tensors."""
    def __init__(self, mean, std):
        self.normalize = Normalize(mean, std)

    def __call__(self, sample):
        sample['t1w'] = self.normalize(sample['t1w'])
        return sample


class Interpolate:
    """Interpolate ndarray."""
    # TODO delete if not needed
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
            sample[key] = ndimage.zoom(sample[key]['data'], zoom=factor, order=0, mode='nearest')
        return sample
