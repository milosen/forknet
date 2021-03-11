import torch
import matplotlib.pyplot as plt
import numpy as np

from forknet.utils.helper import overlay


def load_volume(file: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(file, map_location=device).squeeze().detach().cpu()


def dice(pred, gt, smooth=1e-5):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    n = gt.size(0)
    pred_flat = pred.reshape(n, -1)
    gt_flat = gt.reshape(n, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / n


def show_hist(group1, group2, group3, group_labels, title, ylabel):

    x = np.arange(len(group_labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(x - width, group1, width, label='Saggital')
    rects2 = ax.bar(x, group2, width, label='Coronal')
    rects3 = ax.bar(x + width, group3, width, label='Axial')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 3)),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()


def show_hist_2d(group1, group2, group_labels, title, ylabel):

    x = np.arange(len(group_labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(x - width/2, group1, width, label='ForkNet-Axial')
    rects2 = ax.bar(x + width/2, group2, width, label='ForkNet-2.5D')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 3)),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols))
    return result


if __name__ == '__main__':
    pred_saggital = torch.argmax(torch.cat([load_volume("data/s0_BG.pt").unsqueeze(0), load_volume("data/s0_GM.pt").unsqueeze(0), load_volume("data/s0_WM.pt").unsqueeze(0)], dim=0), dim=0).squeeze().transpose(0, 2)
    pred_coronal = torch.argmax(torch.cat([load_volume("data/s1_BG.pt").unsqueeze(0), load_volume("data/s1_GM.pt").unsqueeze(0), load_volume("data/s1_WM.pt").unsqueeze(0)], dim=0), dim=0).squeeze()
    pred_axial = torch.argmax(torch.cat([load_volume("data/s2_BG.pt").unsqueeze(0), load_volume("data/s2_GM.pt").unsqueeze(0), load_volume("data/s2_WM.pt").unsqueeze(0)], dim=0), dim=0).squeeze().transpose(0, 1)

    gt = load_volume("data/target_WM.pt")
    sl = gt[:, 35].numpy()
    t1w = load_volume("data/t1w_test.pt")
    slt1w = t1w[35].numpy()
    overlay(slt1w, sl)
    plt.imshow(slt1w, cmap='gray', interpolation='none')
    plt.show()

    fuzzy = torch.logical_and((pred_axial != pred_coronal), (pred_axial != pred_saggital))
    fuzzy = torch.logical_and(fuzzy, (pred_coronal != pred_saggital)).float()

    non_axial_majority = torch.logical_and((pred_saggital == pred_coronal), (pred_axial != pred_coronal))

    pred = pred_axial

    pred[non_axial_majority == 1] = pred_saggital[non_axial_majority == 1].clone()
    pred_dir = dict(
        BG=torch.zeros(pred.shape),
        GM=torch.zeros(pred.shape),
        WM=torch.zeros(pred.shape)
    )
    pred_dir['BG'][pred == 0] = 1
    pred_dir['GM'][pred == 1] = 1
    pred_dir['WM'][pred == 2] = 1

    vol = pred_axial[:, 35].numpy()
    plt.imshow(vol, cmap='gray', interpolation='none')
    plt.show()

    vol = pred[:, 35].numpy()
    plt.imshow(vol, cmap='gray', interpolation='none')
    plt.show()

    axial = []
    axial_voted = []
    for label in ['BG', 'GM', 'WM']:
        target = load_volume(f"data/target_{label}.pt")
        axial.append(dice(load_volume(f"data/s2_{label}.pt").transpose(0, 1), target).item())
        axial_voted.append(dice(pred_dir[label], target).item())

    show_hist_2d(axial, axial_voted, ['BG', 'GM', 'WM'], 'Dice Koeffizienten nach Label von ForkNet-Axial und ForkNet-2.5D', 'Dice Koeffizient')

    img_grid = np.array([
        #
        load_volume(f"data/target_BG.pt")[:, 35].numpy(),
        load_volume(f"data/s2_BG.pt").transpose(0, 1)[:, 35].numpy(),
        load_volume(f"data/s2_BG.pt").transpose(0, 1)[:, 35].numpy() -
        load_volume(f"data/target_BG.pt")[:, 35].numpy(),
        pred_dir['BG'][:, 35].numpy(),
        pred_dir['BG'][:, 35].numpy() -
        load_volume(f"data/target_BG.pt")[:, 35].numpy(),
        #
        load_volume(f"data/target_GM.pt")[:, 35].numpy(),
        load_volume(f"data/s2_GM.pt").transpose(0, 1)[:, 35].numpy(),
        load_volume(f"data/s2_GM.pt").transpose(0, 1)[:, 35].numpy() -
        load_volume(f"data/target_GM.pt")[:, 35].numpy(),
        pred_dir['GM'][:, 35].numpy(),
        pred_dir['GM'][:, 35].numpy() -
        load_volume(f"data/target_GM.pt")[:, 35].numpy(),
        #
        load_volume(f"data/target_WM.pt")[:, 35].numpy(),
        load_volume(f"data/s2_WM.pt").transpose(0, 1)[:, 35].numpy(),
        load_volume(f"data/s2_WM.pt").transpose(0, 1)[:, 35].numpy() -
        load_volume(f"data/target_WM.pt")[:, 35].numpy(),
        pred_dir['WM'][:, 35].numpy(),
        pred_dir['WM'][:, 35].numpy() -
        load_volume(f"data/target_WM.pt")[:, 35].numpy(),
    ])

    result = gallery(img_grid, ncols=5)
    cay = plt.matshow(result, interpolation='none')
    plt.colorbar(cay)
    plt.show()
