import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pysmilesutils.tokenize import SMILESAtomTokenizer
import sys
import os
import logging
import time as t
import optuna
import numpy as np

# settings for hpc:
# change wd to current folder to match relative paths
script_dir = os.path.dirname(os.path.abspath(__file__)) # get absolute path of current directory
os.chdir(script_dir)    # change the current directory to the specified directory

# insert top dir to system path for hpc
sys.path.insert(0, '../')

# logging settings
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from utils import (split_SMILE_Dataset, SMILESDataset, train_STP_model, 
                   evaluate_STP_model, RMSELoss,SMILESAugmentation, STP)



def objective(trial, train_dataset, val_dataset, device):
    """
    Objective function for hyperparameter tuning using Optuna.
    This function suggests hyperparameters, trains a model, evaluates it, and reports the validation loss to Optuna.
    It also handles pruning of unpromising trials.
    Args:
        trial (optuna.trial.Trial): A trial object that provides interfaces to suggest hyperparameters.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        float: The final validation loss to be minimized.
    """
    # Suggest hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 6)
    ff_dim = trial.suggest_categorical('ff_dim', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-6)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # Fixed parameters (you can also tune these if needed)
    num_epochs = 30  # Use fewer epochs for faster tuning

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = STP(len(tokenizer), embed_dim, num_heads, ff_dim, num_layers, dropout, max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = RMSELoss()

    val_losses = []
    train_losses = []

    # Training loop with pruning
    for epoch in range(num_epochs):
        train_loss = train_STP_model(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate_STP_model(model, val_loader, loss_fn, device)

        # Report intermediate metric to Optuna
        trial.report(val_loss, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # If the trial should be pruned, raise the prune exception
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # plot and save the loss
    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(f"../deep_learning_outputs/figures/Hypertuning/STP_v7_loss_trial_{trial.number}.png")
        

    # Return the final validation loss to be minimized
    return val_loss


# Training and Evaluation


# Main script
if __name__ == "__main__":
    # Load your CSV file
    logP_df = pd.read_csv("../data/raw/logP.csv", header=0)
    smiles = logP_df['smiles'].tolist()  # Column with SMILES
    targets = logP_df['logP'].tolist()  # Column with logP values

    # Tokenizer and Dataset
    # tokenizer = SMILESTokenizer()
    # tokenizer.fit(smiles)
    tokenizer = SMILESAtomTokenizer(smiles=smiles)
    max_len = 180


    # Split the dataset into training and validation
    dataset = SMILESDataset(smiles, targets, tokenizer, max_length=max_len)
    train_dataset, val_dataset = split_SMILE_Dataset(dataset, ratio=0.8, seed=42)
    train_dataset, test_dataset = split_SMILE_Dataset(train_dataset, ratio=0.8, seed=42)
    

    # Traindata augmentation
    # augmentation = SMILESAugmentation(train_dataset, tokenizer)
    # smiles, targets = augmentation.augment(num_samples=2)
    # train_dataset = SMILESDataset(smiles, targets, tokenizer, max_length=max_len)


    # logging shape infos
    logging.info(f'Train data shape: {len(train_dataset)}')
    logging.info(f'Validation data shape: {len(val_dataset)}')
    logging.info(f'Test data shape: {len(test_dataset)}')

    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    



    
    # study = optuna.create_study(direction='minimize')
    # study.optimize(lambda trial: objective(trial, train_dataset=train_dataset, val_dataset= val_dataset, device=device), n_trials=50)

    # logging.info(f"Best trial: {study.best_trial.value}")

    # # Extract the best hyperparameters
    # best_params = study.best_trial.params

    # logging.info(f"Best hyperparameters: {best_params}")
    # # save best hyperparameters to a text file
    # with open("../deep_learning_outputs/figures/Hypertuning/STP_v7_best_param.txt", "w") as f:
    #     f.write(f"Best hyperparameters: {best_params}")

    # # save all trials to a csv file
    # study.trials_dataframe().to_csv("../deep_learning_outputs/figures/Hypertuning/STP_v7_trials.csv")

    # # Prepare data loaders with the best batch size
    # batch_size = best_params['batch_size']
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    num_epochs = 50

    model = STP(vocab_size = len(tokenizer),
                embed_size = 256, #best_params['embed_dim'], 
                num_heads = 4, #best_params['num_heads'], 
                ff_hidden_dim = 2048, #best_params['ff_dim'], 
                num_layers = 8, #best_params['num_layers'], 
                dropout = 0.15, #best_params['dropout'], 
                max_seq_length = max_len
                ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr= 1e-07)#best_params['learning_rate'])
    loss_fn = RMSELoss()
    
    # Training Loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_STP_model(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate_STP_model(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        

    # Save metrics

    
    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig("../deep_learning_outputs/figures/STP_v7_loss_best_param_06.png")
    

    # Test the model and plot predictions
    _, predictions, true_values = evaluate_STP_model(model, val_loader, loss_fn, device)
    predictions = [pred[0] for pred in predictions]

    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.7)
    plt.xlabel("True Values (logP)")
    plt.ylabel("Predicted Values (logP)")
    plt.title("True vs Predicted Values")
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.savefig("../deep_learning_outputs/figures/STP_v7_predict_best_param_06.png")

    
    # Save the predictions and true values
    npz_file = '../deep_learning_outputs/model_evaluation/STP_v7_06.npz'

    np.savez(
        npz_file,
        y_pred=predictions,
        y_test=true_values
    )
    
    # Save the trained model
    torch.save(model.state_dict(), "../deep_learning_outputs/trained_models/STP_v7_best_param_06.pth")
