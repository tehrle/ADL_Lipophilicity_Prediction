# import necessary packages
import matplotlib.pyplot as plt
import numpy as np

#=======================================================================================================================

#
#   Functions for Plotting PyTorch Stuff
#

def plot_history(file, train_history, val_history, train_label, val_label,
                 train_color='blue', val_color='green', train_linestyles='solid', val_linestyles='solid',
                 figsize=(16, 8), xlabel='Number of epochs / -', ylabel='Loss / -', title=None):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        np.arange(1, len(train_history) + 1),
        train_history,
        label=train_label,
        color=train_color,
        linestyle=train_linestyles
    )
    ax.plot(
        np.arange(1, len(val_history) + 1),
        val_history,
        label=val_label,
        color=val_color,
        linestyle=val_linestyles
    )

    # set basic properties
    ax.grid(False)
    ax.set_facecolor('white')

    # set title, labels and font size
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=14, framealpha=0)
    ax.tick_params(axis='both', labelsize=14)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_edgecolor('black')

    # adjust margins
    ax.margins(x=0)
    fig.tight_layout(pad=0)

    fig.savefig(file)

#=======================================================================================================================

#
#   General Functions for Plotting
#

def plot_logp_correlation(file, y_true, y_pred, xlim, ylim, figsize=(8, 8)):

    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, color='blue', alpha=0.7, edgecolors='k', label='Data Points')
    ax.plot([-20, 20], [-20, 20], color='black', linestyle='--', label='Identity Line (y=x)')
    ax.text(xlim[0] + 1, ylim[1] - 1, f'Pearson Correlation: {pearson_corr:.2f}', fontsize=14)

    # set basic properties
    ax.grid(False)
    ax.set_facecolor('white')

    # set title, labels and font size
    ax.set_title('True vs Predicted logP', fontsize=16)
    ax.set_xlabel('True logP-Value / -', fontsize=14)
    ax.set_ylabel('Predicted logP-Value / -', fontsize=14)
    ax.legend(fontsize=14, framealpha=0)
    ax.tick_params(axis='both', labelsize=14)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_edgecolor('black')

    # Adjust axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # adjust margins
    ax.margins(x=0)
    fig.tight_layout(pad=0)

    fig.savefig(file)