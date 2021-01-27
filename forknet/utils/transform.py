import torch
import kornia.augmentation as k


class Transform(torch.nn.Module):

    def __init__(self, mean, std, scale=(0.9, 1.1), max_degrees=2) -> None:
        super(Transform, self).__init__()
        self.max_degrees = max_degrees
        self.aff = k.RandomAffine(max_degrees, resample=k.Resample.NEAREST, scale=scale)
        self.norm = k.Normalize(mean, std)

    def forward(self, data: dict):
        aff_params = self.aff.generate_parameters(data['t1w'].shape)

        for tissue_mask in data:
            data[tissue_mask] = self.aff(data[tissue_mask], aff_params)

        data['t1w'] = self.norm(data['t1w'])

        return data
