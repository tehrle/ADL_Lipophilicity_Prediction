"""
This module provides custom loss functions / metrics for PyTorch.

Classes:
--------
- RMSELoss: Computes the Root Mean Squared Error (RMSE) between predictions and targets.

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024
"""

# import necessary packages
import torch
import torch.nn as nn
#=======================================================================================================================

class RMSELoss(nn.Module):
    """
    Custom loss function to compute the Root Mean Squared Error (RMSE).

    Attributes:
    -----------
    mse : torch.nn.MSELoss
        An instance of the Mean Squared Error (MSE) loss function used as a basis
        to compute the RMSE.

    Methods:
    --------
    forward(yhat, y):
        Computes the RMSE between predictions (`yhat`) and targets (`y`).
    """
    def __init__(self):
        """
        Initializes the RMSELoss class by setting up the MSE loss function as
        an internal attribute.
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        """
        Computes the Root Mean Squared Error (RMSE) between the predicted and
        target values.

        Parameters:
        -----------
        yhat : torch.Tensor
            The predicted values
        y : torch.Tensor
            The ground truth target values.

        Returns:
        --------
        torch.Tensor
            The RMSE loss value as a scalar tensor.
        """
        return torch.sqrt(self.mse(yhat, y))