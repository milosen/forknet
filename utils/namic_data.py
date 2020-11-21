import os

from typing import Union

import nrrd
import torch
from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np
from torch import randperm


class NAMIC(Dataset):
    """NAMIC dataset."""

    def __init__(self, root_dir: str, transform) -> None:
        """

        :param root_dir:
        :param transform:
        """
        self.transform = transform
        self.root_dir = root_dir
        self.cases = [filename for filename in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        case_number = self.cases[idx][4:]
        filename_seg = os.path.join(
            self.root_dir, self.cases[idx], f'{case_number}-freesurferseg-def.nrrd'
        )
        filename_t1w = os.path.join(
            self.root_dir, self.cases[idx], f'{case_number}-t1w.nrrd'
        )
        seg, header_seg = nrrd.read(filename_seg)
        t1w, header_t1w = nrrd.read(filename_t1w)
        sample = {
            't1w': {'data': t1w, 'header': header_t1w},
            'seg': {'data': seg, 'header': header_seg}
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        seg, t1w = sample['seg'], sample['t1w']
        sample['seg'].update({'data': torch.from_numpy(seg['data'].astype(np.float32))})
        sample['t1w'].update({'data': torch.from_numpy(t1w['data'].astype(np.float32))})
        return sample


class ReverseT1w:
    """Reverse t1w image in axial direction."""
    def __call__(self, sample):
        t1w = sample['t1w']
        sample['t1w'].update({
            'data': np.flip(t1w['data'], axis=2).copy()
        })
        return sample


class Interpolate:
    """Interpolate."""
    def __init__(self, seg_factor: Union[float, tuple],
                 t1w_factor: Union[float, tuple]) -> None:
        self.seg_factor = seg_factor
        self.t1w_factor = t1w_factor

    def __call__(self, sample):
        seg_zoomed = ndimage.zoom(sample['seg']['data'], zoom=self.seg_factor, order=0, mode='nearest')
        t1w_zoomed = ndimage.zoom(sample['t1w']['data'], zoom=self.t1w_factor, order=0, mode='nearest')
        sample['seg'].update({'data': seg_zoomed})
        sample['t1w'].update({'data': t1w_zoomed})
        return sample


class Affine:
    def __call__(self, sample):
        seg0 = np.array([-119.169, -119.169,  71.4])
        t1w0 = np.array([-136.14, -130.248, 29.8626])
        m_seg = np.array([[1.6667, 0., 0.],
                          [0., 1.6667, 0.],
                          [0., 0., -1.700]])
        mres_seg = np.array([[256/144, 0., 0.],
                             [0., 256/144, 0.],
                             [0., 0., 256/85]])
        mres_t1w = np.array([[1, 0., 0.],
                             [0., 1, 0.],
                             [0., 0., 256 / 176]])
        translation = np.array([-8, -12, 92])  # seg0 - np.dot(np.linalg.inv(m_seg), t1w0)
        seg_transformed = ndimage.affine_transform(
            sample['seg']['data'],
            matrix=np.linalg.inv(m_seg),
            offset=translation,
            output_shape=(256, 256, 256), mode='constant', order=0
        )
        t1w_transformed = ndimage.affine_transform(
            sample['t1w']['data'],
            matrix=np.array(np.eye(3)),
            output_shape=(256, 256, 256), mode='constant', order=0
        )
        sample['t1w'].update({'data': t1w_transformed})
        sample['seg'].update({'data': seg_transformed})
        return sample


def collate(batch):
    t1w = batch['t1w']['data']
    seg = batch['seg']['data']
    batch_size = t1w.shape[0]
    slices_per_volume = t1w.shape[1]
    slice_shape = t1w.shape[2:]
    slices_per_batch = batch_size * slices_per_volume
    rand_idx = randperm(slices_per_batch)
    t1w = t1w.permute((0, 3, 2, 1)).reshape((slices_per_batch, 1, slice_shape[0], slice_shape[1]))
    seg = seg.permute((0, 3, 2, 1)).reshape((slices_per_batch, 1, slice_shape[0], slice_shape[1]))
    t1w_transformed = t1w.view(t1w.shape)[rand_idx]
    seg_transformed = seg.view(seg.shape)[rand_idx]
    batch['t1w'].update({'data': t1w_transformed})
    batch['seg'].update({'data': seg_transformed})
    return batch