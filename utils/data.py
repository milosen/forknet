import os

import torch
import torch.utils.data
import numpy as np
import nibabel as nib
from torchvision.transforms import Normalize, Compose


class MICCAI18(torch.utils.data.Dataset):
    """MICCAI18 dataset."""
    def __init__(self, base_dir: str, case_list: list) -> None:
        self.base_dir = base_dir
        self.cases = case_list
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
        self.transform = Compose([
            ToTensor()
            , Norm([np.mean(self.data['t1w'])], [np.std(self.data['t1w'])])
        ])

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

        sample = self.transform(sample)

        return sample


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
