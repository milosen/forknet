import unittest

from torch import ShortTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib as plt

from forknet.model import ForkNet
from utils.namic_data import Affine, NAMIC, ToTensor
from utils.visuals import plot_matrix


class TestForwardPass(unittest.TestCase):
    def test_forward(self):
        transform = transforms.Compose([Affine(), ToTensor()])
        dataset = NAMIC(root_dir='../data/', transform=transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        net = ForkNet(classes=1)
        net.eval()
        for i_batch, sample_batched in enumerate(dataloader):
            out = net(sample_batched['t1w']['data'][:, 100:101, :, :])
            plot_matrix(sample_batched['t1w']['data'][0, 100, :, :], 'Input')
            plot_matrix(out[0, 0, :, :].detach().numpy(), 'Output')


if __name__ == '__main__':
    unittest.main()
