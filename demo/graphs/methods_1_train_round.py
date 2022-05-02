import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    labels = ["baseline",
              "pseudo-masks",
              "dilation",
              "slic",
              "grabcut"]

    means = [32.88,
             48.43,
             48.67,
             47.78,
             49.56]

    stdevs = [0,
              1.60,
              2.68,
              2.46,
              2.18]

    # Build the plot
    fig, ax = plt.subplots()
    x_pos = np.arange(len(means))
    ax.bar(x_pos, means, yerr=stdevs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Segmentation Average Precision (AP)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    # ax.set_title('Methods evaluation comparison')
    ax.yaxis.grid(True)

    for i, v in enumerate(means):
        ax.text(i-0.25, v - 5, str(v), color='blue', fontweight='bold')

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('methods_comparison.png', dpi=300)
    plt.show()
