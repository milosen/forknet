import torch
import matplotlib.pyplot as plt
import numpy as np

from forknet.utils.helper import overlay, load_volume, dice_coefficient, gallery


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


if __name__ == '__main__':
    labels = ['BG', 'GM', 'WM']
    pred_saggital = torch.argmax(torch.cat([
        load_volume(f"data/s0_{label}.pt").unsqueeze(0) for label in labels
    ], dim=0), dim=0).squeeze().transpose(0, 2)
    pred_coronal = torch.argmax(torch.cat([
        load_volume(f"data/s1_{label}.pt").unsqueeze(0) for label in labels
    ], dim=0), dim=0).squeeze()
    pred_axial = torch.argmax(torch.cat([
        load_volume(f"data/s2_{label}.pt").unsqueeze(0) for label in labels
    ], dim=0), dim=0).squeeze().transpose(0, 1)

    t1w = load_volume("data/t1w_test.pt")
    slt1w = t1w[35].numpy()

    fuzzy = torch.logical_and((pred_axial != pred_coronal), (pred_axial != pred_saggital))
    fuzzy = torch.logical_and(fuzzy, (pred_coronal != pred_saggital)).float()
    overlay(slt1w, fuzzy[:, 35].numpy())

    non_axial_majority = torch.logical_and((pred_saggital == pred_coronal), (pred_axial != pred_coronal))
    overlay(slt1w, non_axial_majority[:, 35].numpy())

    pred_majority_vote = pred_axial

    pred_majority_vote[non_axial_majority == 1] = pred_saggital[non_axial_majority == 1].clone()
    pred_dir = dict(
        BG=torch.zeros(pred_majority_vote.shape),
        GM=torch.zeros(pred_majority_vote.shape),
        WM=torch.zeros(pred_majority_vote.shape)
    )
    pred_dir['BG'][pred_majority_vote == 0] = 1
    pred_dir['GM'][pred_majority_vote == 1] = 1
    pred_dir['WM'][pred_majority_vote == 2] = 1

    axial = []
    axial_voted = []
    for label in ['BG', 'GM', 'WM']:
        target = load_volume(f"data/target_{label}.pt")
        axial.append(dice_coefficient(load_volume(f"data/s2_{label}.pt").transpose(0, 1), target).item())
        axial_voted.append(dice_coefficient(pred_dir[label], target).item())

    show_hist_2d(axial, axial_voted, ['BG', 'GM', 'WM'],
                 'Dice Koeffizienten nach Label von ForkNet-Axial und ForkNet-2.5D', 'Dice Koeffizient')

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
