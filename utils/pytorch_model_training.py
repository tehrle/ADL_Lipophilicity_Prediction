import numpy as np
import torch
import torch.nn as nn
from .pytorch_loss_metrics import RMSELoss
from tqdm import tqdm

#=======================================================================================================================
#
#   For graph neural network!
#

def predict_graph_model(model, test_loader, device):
    # Init list for predictions / labels
    test_targets = []
    test_predictions = []

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            # Move data (graph) to device
            data = data.to(device)

            # Make predictions
            test_output = model(data.x, data.edge_index, data.edge_attr, data.batch)

            # Store predictions and targets
            test_targets.append(data.y.detach().cpu())
            test_predictions.append(test_output.detach().cpu())

    # Concat and convert to tensor
    y_pred_tensor = torch.cat(test_predictions)
    y_test_tensor = torch.cat(test_targets)

    return y_pred_tensor, y_test_tensor

def validate_graph_model(model, val_loader, loss_fn, device, mae_loss, mse_loss, rmse_loss):
    # Initialize validation val_metrics
    val_metrics = {
        'loss': [],
        'mae': [],
        'mse': [],
        'rmse': []
    }

    # Switch to evaluation mode
    model.eval()

    # Disable gradient tracking
    with torch.no_grad():
        for data in val_loader:
            # Move data to device
            data = data.to(device)

            # Forward pass
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.y

            # Calculate val_metrics
            val_metrics['loss'].append(loss_fn(output, target))
            val_metrics['mae'].append(mae_loss(output, target))
            val_metrics['mse'].append(mse_loss(output, target))
            val_metrics['rmse'].append(rmse_loss(output, target))

    # Compute mean for validation set
    return (
        torch.stack(val_metrics['loss']).mean().item(),
        torch.stack(val_metrics['mae']).mean().item(),
        torch.stack(val_metrics['mse']).mean().item(),
        torch.stack(val_metrics['rmse']).mean().item()
    )



def train_graph_model(model, num_epochs, optimizer, loss_fn, train_loader, val_loader, device):
    # Tracked Metrics
    metrics = {
        'epoch': np.arange(1, num_epochs + 1),
        'train_loss': [],
        'train_mae': [],
        'train_mse': [],
        'train_rmse': [],
        'val_loss': [],
        'val_mae': [],
        'val_mse': [],
        'val_rmse': []
    }

    # Initialize Metrics
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    rmse_loss = RMSELoss()

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_metrics = {
            'loss': [],
            'mae': [],
            'mse': [],
            'rmse': []
        }

        # Progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True, colour='white')

        for data in train_bar:
            # Move data to device
            data = data.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.y  # Extract labels

            # Calculate loss
            loss = loss_fn(output, target)
            train_metrics['loss'].append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_metrics['mae'].append(mae_loss(output, target))
            train_metrics['mse'].append(mse_loss(output, target))
            train_metrics['rmse'].append(rmse_loss(output, target))

            # Update progress bar
            train_bar.set_postfix({
                'train_loss': f'{torch.stack(train_metrics["loss"]).mean().item():.4f}',
                'train_mae': f'{torch.stack(train_metrics["mae"]).mean().item():.4f}',
                'train_mse': f'{torch.stack(train_metrics["mse"]).mean().item():.4f}',
                'train_rmse': f'{torch.stack(train_metrics["rmse"]).mean().item():.4f}',
            })

        # Store metrics for training
        for key in train_metrics:
            metrics[f'train_{key}'].append(torch.stack(train_metrics[key]).mean().item())

        # Validation Model
        val_loss, val_mae, val_mse, val_rmse = validate_graph_model(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            mae_loss=mae_loss,
            mse_loss=mse_loss,
            rmse_loss=rmse_loss,
            device=device
        )

        # Store metrics for validation
        metrics['val_loss'].append(val_loss)
        metrics['val_mae'].append(val_mae)
        metrics['val_mse'].append(val_mse)
        metrics['val_rmse'].append(val_rmse)

    return metrics

#=======================================================================================================================
#  For transformer model
#

def train_STP_model(model, dataloader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_STP_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    return total_loss / len(dataloader), predictions, targets_list

#=======================================================================================================================
