import numpy as np
from matplotlib import pyplot as plt
import statistics

if __name__ == "__main__":
    aps_none = [48.329,	48.141,	45.311,	41.942,	50.976,	42.951,	46.611, 43.21, 45.731]
    # stds_none = [1.140, 1.498, 1.556, 2.236, 1.781, 1.552, 1.399, 1.514, 1.097, 1.734]

    aps_dilation = [48.575,	53.28,	49.65,	50.251,	49.107,	48.948,	46.158,	47.092,	46.6]
    # stds_dilation = [0.692, 1.081, 0.912, 0.446, 1.222, 1.090, 1.279, 1.321, 0.743, 0.872]

    aps_slic = [50.527,	47.585,	47.438,	44.942,	45.713,	43.446,	45.335,	47.357,	47.103]
    # stds_slic = [0.857, 0.851, 0.967, 0.573, 1.676, 2.577, 2.159, 2.201, 1.483, 1.833]

    aps_grabcut = [49.819, 52.047, 53.255,	49.79,	51.4,	46.791,	49.773,	49.042,	47.676]
    # stds_grabcut = [1.685, 0.716, 0.120, 1.400, 0.387, 0.455, 0.765, 0.907, 0.923, 0.652]

    train_rounds = np.arange(start=1, stop=10)
    fig, ax = plt.subplots()
    ax.plot(train_rounds, aps_none, label='Pseudo-masks')
    ax.plot(train_rounds, aps_dilation, label='Dilation method')
    ax.plot(train_rounds, aps_slic, label='Slic method')
    ax.plot(train_rounds, aps_grabcut, label='Grabcut method')
    # ax.errorbar(train_rounds, aps_none, None, capsize=5, marker='d', label='Pseudo-masks')
    # ax.errorbar(train_rounds, aps_dilation, None, capsize=5, marker='d', label='Dilation method')
    # ax.errorbar(train_rounds, aps_slic, None, capsize=5, marker='d', label='Slic method')
    # ax.errorbar(train_rounds, aps_grabcut, None, capsize=5, marker='d', label='Grabcut method')
    ax.set_xticks(train_rounds)
    ax.set_ylabel('Segmentation Average Precision (AP)')
    ax.set_xlabel('Training rounds')
    ax.grid(axis='y')

    ax.legend()
    plt.tight_layout()
    plt.savefig('methods_iterative_comparison.png', dpi=300)
    plt.show()
