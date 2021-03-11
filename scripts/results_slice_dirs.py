import torch
import matplotlib.pyplot as plt
import numpy as np

from forknet.utils.helper import overlay, load_volume, dice_coefficient, gallery


def show_hist(group1, group2, group3, group_labels, title, ylabel):

    x = np.arange(len(group_labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(x - width, group1, width, label='Sagittal')
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


if __name__ == '__main__':
    labels = ['BG', 'GM', 'WM']
    axial_slice = 35

    targets = [load_volume(f"data/target_{label}.pt") for label in labels]
    axials = [load_volume(f"data/s2_{label}.pt").transpose(0, 1) for label in labels]
    coronals = [load_volume(f"data/s1_{label}.pt") for label in labels]
    sagitals = [load_volume(f"data/s0_{label}.pt").transpose(0, 2) for label in labels]

    for metric in [dice_coefficient, torch.nn.functional.binary_cross_entropy]:
        show_hist(
            [metric(sagital, target).item() for sagital, target in zip(sagitals, targets)],
            [metric(coronal, target).item() for coronal, target in zip(coronals, targets)],
            [metric(axial, target).item() for axial, target in zip(axials, targets)],
            group_labels=labels,
            title=f'{metric.__name__} nach Label und Schnittrichtung',
            ylabel=f'{metric.__name__}'
        )

    # sanity check
    for i in range(3):
        overlay(sagitals[i][:, axial_slice], targets[i][:, axial_slice])
        overlay(coronals[i][:, axial_slice], targets[i][:, axial_slice])
        overlay(axials[i][:, axial_slice], targets[i][:, axial_slice])
