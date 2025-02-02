import matplotlib.pyplot as plt
from constants import INITIAL_CUTOFF, TOP


def plot_accuracies_(accuracies):
    plt.hist([[acc for acc, _ in accuracies_] for accuracies_ in accuracies.values()], label=[*accuracies.keys()])
    plt.legend(loc='upper right')
    plt.show()
