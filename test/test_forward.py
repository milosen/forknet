import logging
import unittest

from torchvision import transforms
from torch.utils.data import DataLoader

from forknet.model import ForkNet
from utils.data import Affine, NAMIC, ToTensor, collate
from utils.visuals import plot_matrix


batch_size = 2


class TestForwardPass(unittest.TestCase):
    def test_forward(self):
        logging.basicConfig(level=logging.DEBUG)
        transform = transforms.Compose([Affine(), ToTensor()])
        dataset = NAMIC(root_dir='../data/', transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        net = ForkNet(n_classes=1)
        net.eval()
        logging.basicConfig(level=logging.INFO)
        for i_batch, sample_batched in enumerate(dataloader):
            batch = collate(batch=sample_batched)
            out = net(batch['t1w']['data'])
            plot_matrix(batch['t1w']['data'][200, 0, :, :], 'Input')
            plot_matrix(out[200, 0, :, :].detach().numpy(), 'Output')
            plot_matrix(batch['seg']['data'][200, 0, :, :], 'Input')


if __name__ == '__main__':
    unittest.main()
