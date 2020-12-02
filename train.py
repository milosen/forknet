from datetime import datetime
import os

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch
import torchvision
from tqdm import tqdm

from forknet.model import ForkNet
from utils.data import NAMIC, ToTensor


batch_slices = 20
epochs = 20
lr = 0.01
n_train = 175
n_val = 1
device = "cuda"


def train_net():
    transform = transforms.Compose([
        ToTensor(device=device)
    ])
    dataset = NAMIC(transform=transform)
    net = ForkNet(n_classes=2)
    net.to(device)
    writer = SummaryWriter(os.path.join('runs', datetime.now().strftime("%Y-%m-%d_%H-%M")))
    train, val = random_split(dataset, [n_train, n_val])
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(train, batch_size=batch_slices, shuffle=True, num_workers=0)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    sample_val = iter(val_loader).next()
    inp_val = sample_val.pop('t1w')
    del val_loader
    for e in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {e+1}/{epochs}') as pbar:
            for batch_dict in train_loader:
                inp = batch_dict.pop('t1w')
                out = net(inp['data'])
                losses = [
                    criterion(out[i][:, 0, :, :], batch_dict[seg]['data'][:, 0, :, :])
                    for i, seg in enumerate(batch_dict)
                ]
                torch.autograd.backward(losses)
                optimizer.step()
                pbar.update(inp['data'].shape[0])
                del inp, out
            with torch.no_grad():
                out = net(inp_val['data'])
                # prepare losses
                losses = [
                    criterion(out[i][:, 0, :, :], sample_val[seg]['data'][:, 0, :, :])
                    for i, seg in enumerate(sample_val)
                ]
                for i, loss in enumerate(losses):
                    writer.add_scalar(f'loss{i}', loss, global_step=e)
                # image grid of validation sample
                img_grid = [inp_val['data'][0]]
                for i, seg in enumerate(sample_val):
                    img_grid.append(sample_val[seg]['data'][0])
                    img_grid.append(out[i][0])
                writer.add_image(
                    img_tensor=torchvision.utils.make_grid(img_grid),
                    tag='data (input, ref_gm, out_gm, ref_wm , out_wm)',
                    global_step=e
                )
                if e == epochs-1:
                    writer.add_graph(net, inp_val['data'])
    writer.flush()
    writer.close()


if __name__ == '__main__':
    train_net()
