"""
This module provides functions to visualize training and validation loss histories and analyze correlations between
predicted and true logP values.

Functions:
-----------
- plot_history: Plots training and validation loss history over epochs.
- plot_logp_correlation: Visualizes the correlation between true and predicted logP values.

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024
"""

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
    """
    This function generates a plot comparing the training and validation losses across
    all epochs.

    Parameters:
    -----------
    file : str
        The file path where the plot will be saved.
    train_history : list or np.ndarray
        The training loss values for each epoch.
    val_history : list or np.ndarray
        The validation loss values for each epoch.
    train_label : str
        The label for the training loss curve.
    val_label : str
        The label for the validation loss curve.
    train_color : str, optional
        The color of the training loss curve. Defaults to 'blue'.
    val_color : str, optional
        The color of the validation loss curve. Defaults to 'green'.
    train_linestyles : str, optional
        The line style of the training loss curve. Defaults to 'solid'.
    val_linestyles : str, optional
        The line style of the validation loss curve. Defaults to 'solid'.
    figsize : tuple, optional
        The size of the figure. Defaults to (16, 8).
    xlabel : str, optional
        The label for the x-axis. Defaults to 'Number of epochs / -'.
    ylabel : str, optional
        The label for the y-axis. Defaults to 'Loss / -'.
    title : str, optional
        The title of the plot. Defaults to None.

    Returns
    -------
    None
    """

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
    """
    This function creates a scatter plot of true vs. predicted logP values,
    includes a dashed identity line (y=x), and annotates the plot with the Pearson correlation coefficient.

    Parameters:
    -----------
    file : str
        The file path where the plot will be saved.
    y_true : list or np.ndarray
        Array of true logP values.
    y_pred : list or np.ndarray
        Array of predicted logP values.
    xlim : tuple
        The x-axis limits specified as (min, max).
    ylim : tuple
        The y-axis limits specified as (min, max).
    figsize : tuple, optional
        The size of the figure. Defaults to (8, 8).

    Returns
    -------
    None
    """
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