import numpy as np
from matplotlib import pyplot as plt
import statistics

if __name__ == "__main__":
    aps = [59.444, 61.489, 60.71, 61.617, 61.912, 62.185, 60.211, 60.313, 61.971, 62.062]
    train_rounds = np.arange(start=1, stop=11)
    print(train_rounds)
    plt.plot(train_rounds, aps, marker='o', scaley=1)
    plt.ylim((50, 65))
    plt.show()
