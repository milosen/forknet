import unittest

from torchvision import transforms

from forknet.utils.data import NAMIC, Affine
from forknet.utils.helper import overlay


class TestData(unittest.TestCase):
    def test_data(self):
        transform = transforms.Compose([
            Affine(),
            # Interpolate(seg_factor=(256./144, 256./144, 256./85),
            #             t1w_factor=(1, 1, 256./176))
        ])
        dataset = NAMIC(root_dir='../data/', transform=transform)
        d = dataset[0]
        overlay(
            d['t1w']['data'][:, :, 100],
            d['seg']['data'][:, :, 100],
            'Overlay'
        )
        print(d['t1w']['header'])


if __name__ == '__main__':
    unittest.main()
