import torch
import torch.utils.data
import torchvision
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

from forknet.datasets.miccai18 import MICCAI18
from forknet.train import init_forknet, validate_net, generate_img_grid, slice_all
from forknet.utils.helper import plot_matrix


def plot_roc_curve(pred: torch.Tensor, label: torch.Tensor, name="example estimator"):
    pred, label = pred.detach().cpu().numpy().flatten(), label.detach().cpu().numpy().flatten()
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name=name)

    # calc optimal threshold
    idx = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=idx), 'threshold': pd.Series(thresholds, index=idx)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    display.plot()
    plt.show()

    return list(roc_t['threshold'])[0]


def test_net(load, xslice, show, test=False):
    if test:
        test_case = ['1']  # test case
    else:
        test_case = ['4']  # validation case
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MICCAI18(base_dir='data/miccai18/training', case_list=test_case)
    net = init_forknet(load=load, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    test_batch_dict = iter(test_loader).next()
    for tissue in test_batch_dict:
        test_batch_dict[tissue] = test_batch_dict[tissue].to(device)
    losses, outputs = validate_net(net=net, batch_dict=test_batch_dict, criterion=criterion)
    logging.info(f"Losses: {losses}")
    if show:
        plot_matrix(
            torchvision.utils.make_grid(
                generate_img_grid(test_batch_dict, net, xslice),
                nrow=len(dataset.tissues) + 1
            ).detach().cpu().numpy()[0]
        )
        thresholds = []
        for tissue in MICCAI18.tissues:
            best_threshold = plot_roc_curve(
                name=f"ROC curve for {tissue} tissue",
                label=test_batch_dict[tissue],
                pred=outputs[tissue]
            )
            logging.info(f"Best threshold: {best_threshold}")
            thresholds.append(best_threshold)
        t_losses, t_outputs = validate_net(net=net, batch_dict=test_batch_dict, criterion=criterion,
                                           thresholds=thresholds)
        logging.info(f"Losses with threshold: {t_losses}")
        plot_matrix(
            torchvision.utils.make_grid(
                generate_img_grid(test_batch_dict, net, xslice, thresholds),
                nrow=len(dataset.tissues) + 1
            ).detach().cpu().numpy()[0]
        )
