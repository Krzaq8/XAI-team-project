import matplotlib.pyplot as plt
from constants import INITIAL_CUTOFF, TOP


def plot_accuracies_(accuracies, baseline):
    colors = ['red', 'green', 'blue']
    plt.hist([[acc for acc, _ in accuracies_] for accuracies_ in accuracies.values()], label=[*accuracies.keys()],
             color=colors, bins=30)
    plt.axvline(baseline, color='black', linestyle='-', linewidth=1, label='baseline')
    thresholds = [sorted([e for e in accuracies_ if e[0] != 'nan'], key=lambda x: x[0])[-int(TOP*len(accuracies_))][0] for accuracies_ in accuracies.values()]
    labels = accuracies.keys()
    for color, threshold, label in zip(colors, thresholds, labels):
        plt.axvline(threshold, color=color, linestyle='--', linewidth=1)
    plt.legend(loc='upper left')
    plt.xlabel('accuracy')
    plt.ylabel('count')
    plt.show()
