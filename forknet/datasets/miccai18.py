import os

import torch
import torch.utils.data
import numpy as np
import nibabel as nib
from torchvision.transforms import Compose


from forknet.utils.data import ToTensor


class MICCAI18(torch.utils.data.Dataset):
    """MICCAI18 dataset."""
    z_slices = 64
    img_dims = (256, 256)
    labels = dict(BG=0, GM=1, WM=3)
    # we will need ordered tissue types
    tissues = [x[0] for x in sorted(labels.items())]

    def __init__(self, base_dir: str, case_list: list, train: bool = True,
                 slice_axis: int = 0) -> None:
        self.base_dir = base_dir
        self.cases = case_list
        self.train = train
        self.slice_axis = slice_axis

        if self.slice_axis == 2:
            self.sliced_img_dims = MICCAI18.img_dims
            self.n_slices = MICCAI18.z_slices
            self.change_dims = (2, 1, 0)
        else:
            self.sliced_img_dims = (MICCAI18.img_dims[1-self.slice_axis], MICCAI18.z_slices)
            self.n_slices = MICCAI18.img_dims[self.slice_axis]
            if self.slice_axis == 1:
                self.change_dims = (1, 2, 0)
            else:
                self.change_dims = (0, 2, 1)

        self.n_datapoints = len(self.cases) * self.n_slices

        # stack all the segmented axial slices from all subjects
        seg = np.concatenate(tuple(
            np.flip(np.pad(
                np.array(nib.load(
                    os.path.join(self.base_dir, case, 'segm.nii.gz')
                ).dataobj), ((8, 8), (8, 8), (8, 8))
            ), axis=2) for case in self.cases
        ), axis=self.slice_axis)
        self.data = dict(
            WM=np.zeros(seg.shape),
            GM=np.zeros(seg.shape),
            BG=np.ones(seg.shape),
            t1w=np.concatenate(tuple(
                np.flip(np.pad(
                    np.array(nib.load(
                        os.path.join(self.base_dir, case, 'orig', 'reg_T1.nii.gz')
                    ).dataobj), ((8, 8), (8, 8), (8, 8))
                ), axis=2) for case in self.cases
            ), axis=self.slice_axis)
        )
        for mod in self.data:
            self.data[mod] = self.data[mod].transpose(self.change_dims)
        seg = seg.transpose(self.change_dims)
        # create binary label maps
        for tissue in MICCAI18.tissues:
            label = MICCAI18.labels[tissue]
            if tissue != 'BG':
                self.data[tissue][seg == label] = 1
                self.data['BG'][seg == label] = 0

        self.mean = torch.from_numpy(np.array([np.mean(self.data['t1w'])]))
        self.std = torch.from_numpy(np.array([np.std(self.data['t1w'])]))
        self.init_transform = Compose([
            ToTensor()
        ])
        self.data = self.init_transform(self.data)

    def __len__(self):
        return self.n_datapoints

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.get_slice(idx)

    def get_slice(self, idx: int) -> dict:
        sample = {
            mod: self.data[mod].index_select(0, torch.LongTensor([idx])) for mod in self.data
        }

        return sample
