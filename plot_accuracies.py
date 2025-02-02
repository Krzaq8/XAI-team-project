import matplotlib.pyplot as plt
from constants import INITIAL_CUTOFF, TOP


def plot_accuracies_(accuracies, baseline):
    plt.hist([[acc for acc, _ in accuracies_] for accuracies_ in accuracies.values()], label=[*accuracies.keys()])
    plt.axvline(baseline, color='black', linestyle='-', linewidth=1, label='baseline')
    colors = ['red', 'green', 'blue']
    thresholds = [sorted(accuracies_, key=lambda x: x[0])[-int(TOP*len(accuracies_))] for accuracies_ in accuracies.values()]
    labels = accuracies.keys()
    for color, threshold, label in zip(colors, thresholds, labels):
        plt.axvline(threshold, color=color, linestyle='--', linewidth=1, label=label)
    plt.legend(loc='upper right')
    plt.show()
