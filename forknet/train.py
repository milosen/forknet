import logging
import sys
import contextlib
import os

import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from tqdm import tqdm
import ray.tune
import numpy as np

from forknet.model import ForkNet
from forknet.utils.transform import Transform
from forknet.datasets.miccai18 import MICCAI18


torch.backends.cudnn.benchmark = False
torch.manual_seed(10)
np.random.seed(10)
writer = torch.utils.tensorboard.SummaryWriter()


def init_forknet(load, device):
    net = ForkNet(tissues=MICCAI18.tissues)
    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Loaded model from {load}')
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    return net


def norm_img_tensor(img: torch.Tensor):
    return (img - img.min()) / torch.max(img)


def generate_img_grid(val_batch_dict, net, xslice=20, thresholds=None, transform=None):
    val_batch_dict = dict(val_batch_dict)
    sliced_batch_dict = slice_all(val_batch_dict, xslice=xslice)
    if transform:
        # just for debugging
        sliced_batch_dict = transform(sliced_batch_dict)
    input_slice = sliced_batch_dict.pop('t1w')
    output_slices_dict = net(input_slice)
    input_slice = input_slice[0]
    device = input_slice.device
    masks = [sliced_batch_dict[tissue][0] for tissue in MICCAI18.tissues]
    output_slices = [torch.sigmoid(output_slices_dict[tissue][0]) for tissue in MICCAI18.tissues]
    img_grid = [norm_img_tensor(input_slice)]
    for target_mask in masks:
        img_grid.append(target_mask)
    img_grid.append(torch.zeros((1, 256, 256), device=device))
    for output_slice in output_slices:
        img_grid.append(output_slice)
    if thresholds:
        img_grid.append(torch.zeros((1, 256, 256), device=device))
        for i, output_slice in enumerate(output_slices):
            img_grid.append((output_slice > thresholds[i]).float())

    return img_grid


def slice_all(batch_dict, xslice=20):
    for tensor in batch_dict:
        batch_dict[tensor] = batch_dict[tensor][xslice].unsqueeze(dim=0)

    return batch_dict


def threshold_forknet_output(forknet_output_tensors: torch.Tensor):
    return tuple(
        (out >= 0.5).float()
        for out in forknet_output_tensors
    )


def validate_net(net,
                 batch_dict,
                 criterion,
                 thresholds=None):
    with torch.no_grad():
        batch_dict = dict(batch_dict)
        validation_input_tensor = batch_dict.pop('t1w')
        validation_output = net(validation_input_tensor)
        for tissue in validation_output:
            validation_output[tissue] = torch.sigmoid(validation_output[tissue])
        tissue_losses = []
        for i, tissue in enumerate(validation_output):
            if thresholds:
                output = (validation_output[tissue] >= thresholds[i]).float()
            else:
                output = validation_output[tissue]
            loss = criterion(
                output,
                batch_dict[tissue]
            )
            tissue_losses.append(loss)

        return tissue_losses, validation_output


def tensorboard_write(global_step,
                      losses,
                      val_losses,
                      val_batch_dict,
                      net,
                      write_network_graph=False,
                      transform=None):
    # TODO: tidy up
    for i, tissue in enumerate(MICCAI18.tissues):
        writer.add_scalars(
            f'loss/{tissue}', {
                'train': losses[i].item(),
                'validation': val_losses[i].item()
            },
            global_step=global_step
        )
    writer.add_image(
        img_tensor=torchvision.utils.make_grid(
            generate_img_grid(val_batch_dict, net, transform=transform),
            nrow=len(MICCAI18.tissues) + 1
        ),
        tag=f'validation sample {MICCAI18.tissues}',
        global_step=global_step
    )
    gradients = torch.cat([param.grad.view(-1) for param in net.parameters()])
    writer.add_histogram(f'Gradients', gradients, global_step=global_step)
    if write_network_graph and global_step == 0:
        writer.add_graph(net, torch.zeros((1, 1, 256, 256), device=val_batch_dict['t1w'].device))


def train_net(batch_size,
              epochs,
              lr,
              betas,
              eps,
              l2_penalty,
              amsgrad,
              load=None,
              checkpoint=None,
              use_raytune=False,
              data_dir='data/miccai18/training'):
    # test_case = ['1']
    validation_cases = ['4']
    train_cases = ['5', '7', '14', '070', '148']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MICCAI18(base_dir=data_dir, case_list=train_cases)
    val_dataset = MICCAI18(base_dir=data_dir, case_list=validation_cases)

    # the validation set consists of just one batch
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset),
                                             shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    net = ForkNet(tissues=MICCAI18.tissues)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=l2_penalty, amsgrad=amsgrad
    )
    if load:
        state = torch.load(load, map_location=device)
        if state is tuple:
            model_state, optimizer_state = state
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            logging.info(f'Loaded model and optimizer from {load}')
        else:
            net.load_state_dict(state)
            logging.info(f'Loaded model from {load}')
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        logging.info(f"Using DataParallel")
    net.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    transform = Transform(mean=train_dataset.mean, std=train_dataset.std)

    if not use_raytune:
        logging.info(f'Network:\n'
                     f'\t{net.n_classes} decoder tracks: {train_dataset.tissues}')
        logging.info(f'''Training:
                Epochs:             {epochs}
                Batch size:         {batch_size}
                Learning rate:      {lr}
                Betas:              {betas}
                Eps:                {eps}
                L2 Penalty:         {l2_penalty}
                Using AMSGrad:      {amsgrad}
                Device:             {device.type}
            ''')

    try:
        for epoch in range(epochs):
            if use_raytune:
                maybe_bar = contextlib.nullcontext()
            else:
                maybe_bar = tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit=' slices')
            with maybe_bar as bar:
                for batch_dict in train_loader:
                    for tensor in batch_dict:
                        batch_dict[tensor] = batch_dict[tensor].to(device)
                    batch_dict = transform(batch_dict)
                    inp = batch_dict.pop('t1w')
                    out = net(inp)
                    losses = [
                        criterion(
                            out[tissue],
                            batch_dict[tissue]
                        ) for tissue in MICCAI18.tissues
                    ]
                    optimizer.zero_grad()
                    torch.autograd.backward(losses)
                    optimizer.step()
                    bar.update(inp.shape[0]) if not use_raytune else None
            validation_batch_dict = iter(val_loader).next()
            for tensor in validation_batch_dict:
                validation_batch_dict[tensor] = validation_batch_dict[tensor].to(device)
            val_losses, _ = validate_net(net=net, batch_dict=validation_batch_dict, criterion=criterion)
            if use_raytune:
                with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((net.state_dict(), optimizer.state_dict()), path)
                ray.tune.report(loss=sum([loss.item() for loss in val_losses]))
            else:
                if checkpoint and (epoch % checkpoint) == (checkpoint - 1):
                    torch.save(net.state_dict(), 'scheduled_checkpoint.pth')
                    logging.info(f'Saved checkpoint for epoch {epoch + 1}')
                tensorboard_write(
                    net=net, global_step=epoch,
                    losses=losses, val_losses=val_losses,
                    val_batch_dict=validation_batch_dict,
                    transform=transform
                )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'interrupt.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
    finally:
        writer.flush()
        writer.close()


def print_network_and_exit(load):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(init_forknet(load, device))
