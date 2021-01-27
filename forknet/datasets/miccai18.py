import os

import torch
import torch.utils.data
import numpy as np
import nibabel as nib
from torchvision.transforms import Compose

from forknet.utils.data import ToTensor, Norm


class MICCAI18(torch.utils.data.Dataset):
    """MICCAI18 dataset."""
    n_slices = 48
    img_dims = (256, 256)
    labels = dict(GM=1, WM=3)
    # we will need ordered tissue types
    tissues = [x[0] for x in sorted(labels.items())]

    def __init__(self, base_dir: str, case_list: list, train: bool = True) -> None:
        self.base_dir = base_dir
        self.cases = case_list
        self.train = train

        # stack all the segmented axial slices from all subjects
        seg = np.concatenate(tuple(
            np.pad(
                np.array(nib.load(
                    os.path.join(self.base_dir, case, 'segm.nii.gz')
                ).dataobj), ((8, 8), (8, 8), (0, 0))
            ) for case in self.cases
        ), axis=2)
        self.data = dict(
            WM=np.zeros((MICCAI18.img_dims[0], MICCAI18.img_dims[1], MICCAI18.n_slices * len(self.cases))),
            GM=np.zeros((MICCAI18.img_dims[0], MICCAI18.img_dims[1], MICCAI18.n_slices * len(self.cases))),
            t1w=np.concatenate(tuple(
                np.pad(
                    np.array(nib.load(
                        os.path.join(self.base_dir, case, 'orig', 'reg_T1.nii.gz')
                    ).dataobj), ((8, 8), (8, 8), (0, 0))
                ) for case in self.cases
            ), axis=2)
        )
        # create binary label maps
        for tissue in MICCAI18.tissues:
            label = MICCAI18.labels[tissue]
            self.data[tissue][seg == label] = 1
        self.mean = torch.from_numpy(np.array([np.mean(self.data['t1w'])]))
        self.std = torch.from_numpy(np.array([np.std(self.data['t1w'])]))
        self.init_transform = Compose([
            ToTensor()
        ])
        self.data = self.init_transform(self.data)

    def __len__(self):
        return len(self.cases)*MICCAI18.n_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.get_slice(idx)

    def get_slice(self, idx):
        sample = {
            mod: self.data[mod][:, :, idx].reshape(
                    # prepend sample dimension for batching
                    (1, MICCAI18.img_dims[0], MICCAI18.img_dims[1])
                ) for mod in self.data
        }

        return sample
