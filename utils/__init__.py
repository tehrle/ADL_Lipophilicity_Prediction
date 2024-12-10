"""
A package containing workflows for processing SMILES strings, SMILES augmentation, convert SMILES into graph objects
and pytorch functions / classes used for deep learning workflows.

This package provides a range of tools for:
- Processing SMILES strings and converting them into molecular graphs or descriptors.
- SMILES data augmentation for expanding datasets.
- Custom PyTorch layers and models for graph neural networks (GNNs) and transformer-based architectures.
- Training, evaluation, and hyperparameter tuning for deep learning models.
- Loss metrics for regression tasks, including RMSE.
- Visualization of training histories, correlation analyses, and performance metrics.

Version:
--------
0.1.0

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024

Modules:
--------
- pytorch_model_training: Functions for training, evaluating, and hyperparameter tuning.
- pytorch_data_handling: Utilities for handling and splitting datasets.
- pytorch_loss_metrics: Custom loss metrics for regression tasks.
- plotting: Functions for visualizing training metrics and correlation plots.
- smiles_processing: Tools for processing SMILES data, including dataset and augmentation utilities.
- pytorch_custom_layers: Custom PyTorch layers and models for advanced architectures.
"""

__version__ = '0.1.0'
__author__ = 'Timo Ehrle & Levin Willi'
#=======================================================================================================================

from .pytorch_model_training import (
    train_graph_model,
    validate_graph_model,
    predict_graph_model,
    train_STP_model,
    evaluate_STP_model,
    train_STP,
    evaluate_STP,
    objective
)

from .pytorch_data_handling import (
    split_dataset,
    split_SMILE_Dataset,
    split_SMILES
)

from .pytorch_loss_metrics import (
    RMSELoss
)

from .plotting import (
    plot_history,
    plot_logp_correlation
)

from .smiles_processing import (
    SMILESConverter,  
    SMILESDataset,
    SMILESAugmentation
)

from .pytorch_custom_layers import (
    TransformerBlock,
    LogPPredictionModel
)