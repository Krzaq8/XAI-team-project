import matplotlib.pyplot as plt
from constants import INITIAL_CUTOFF, TOP


def plot_accuracies_(accuracies):
    plt.hist([[acc for acc, _ in accuracies_] for accuracies_ in accuracies.values()], label=[*accuracies.keys()])
    plt.axvline(BASELINE, color='green', linestyle='-', linewidth=1, label='baseline')
    plt.legend(loc='upper right')
    plt.show()
