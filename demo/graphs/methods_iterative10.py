import numpy as np
from matplotlib import pyplot as plt
import statistics

if __name__ == "__main__":
    aps_none = [58.633, 58.215, 58.563, 57.456, 57.533, 56.657, 56.886, 56.927, 56.972, 57.364]
    stds_none = [1.140, 1.498, 1.556, 2.236, 1.781, 1.552, 1.399, 1.514, 1.097, 1.734]

    aps_dilation = [60.682, 60.311, 59.271, 57.531, 56.608, 56.875, 55.842, 55.924, 54.839, 54.460]
    stds_dilation = [0.692, 1.081, 0.912, 0.446, 1.222, 1.090, 1.279, 1.321, 0.743, 0.872]

    aps_slic = [58.721, 58.307, 57.470, 57.299, 55.964, 56.205, 55.666, 56.652, 55.845, 55.166]
    stds_slic = [0.857, 0.851, 0.967, 0.573, 1.676, 2.577, 2.159, 2.201, 1.483, 1.833]

    aps_grabcut = [61.057, 61.081, 60.871, 61.305, 61.412, 61.543, 61.095, 61.475, 61.374, 61.460]
    stds_grabcut = [1.685, 0.716, 0.120, 1.400, 0.387, 0.455, 0.765, 0.907, 0.923, 0.652]

    train_rounds = np.arange(start=1, stop=11)
    fig, ax = plt.subplots()
    ax.errorbar(train_rounds, aps_none, None, capsize=5, marker='d', label='Pseudo-masks')
    ax.errorbar(train_rounds, aps_dilation, None, capsize=5, marker='d', label='Dilation method')
    ax.errorbar(train_rounds, aps_slic, None, capsize=5, marker='d', label='Slic method')
    ax.errorbar(train_rounds, aps_grabcut, None, capsize=5, marker='d', label='Grabcut method')
    ax.set_xticks(train_rounds)
    ax.set_ylabel('Segmentation Average Precision (AP)')
    ax.set_xlabel('Training rounds')
    ax.grid(axis='y')

    ax.legend()
    plt.tight_layout()
    plt.savefig('methods_iterative_comparison.png', dpi=300)
    plt.show()
