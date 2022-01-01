import numpy as np
from matplotlib import pyplot as plt
import statistics

if __name__ == "__main__":
    labels = ["baseline\nw/o crop", "baseline",
              "pseudo-masks", "pseudo-masks\nfinetune",
              "dilation", "dilation\nfinetune",
              "slic (1)", "slic (1)\nfinetune",
              "slic (2)", "slic (2)\nfinetune",
              "slic (3)", "slic (3)\nfinetune",
              "grabcut", "grabcut\nfinetune"]

    means = [5.962, 39.164,
             58.439, 57.645,
             59.786, 58.229,
             58.439, 56.273,
             59.401, 57.997,
             59.178, 57.093,
             60.424, 58.656]

    stdevs = [0, 0,
              1.301, 0.965,
              1.337, 0.527,
              0.824, 0.445,
              0.735, 1.310,
              1.094, 1.299,
              1.538, 1.227]

    # means = [5.962, 58.439, 59.786, 58.439, 0, 59.178, 60.424]
    # means_finetune = [39.164, 57.645, 58.229, 56.273, 0, 57.093, 58.656]
    # plt.errorbar(names, means, stdevs, linestyle='None', marker='d')
    # plt.show()

    # Build the plot
    fig, ax = plt.subplots()
    x_pos = np.arange(len(means))
    ax.bar(x_pos, means, yerr=stdevs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Segmentation Average Precision (AP)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    # ax.set_title('Methods evaluation comparison')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
