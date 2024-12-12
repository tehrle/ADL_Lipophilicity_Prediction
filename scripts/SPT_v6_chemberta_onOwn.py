import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import sys
import os
import logging
import time as t
from tqdm import tqdm
import optuna

from transformers import AutoTokenizer, AutoModel

# settings for hpc:
# change wd to current folder to match relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # get absolute path of current directory
os.chdir(script_dir)  # change the current directory to the specified directory

# insert top dir to system path for hpc
sys.path.insert(0, '../')

# logging settings
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from utils import RMSELoss, objective, LogPPredictionModel, train_STP, evaluate_STP

# Load the ChemBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initalize Data
logP_df = pd.read_csv("../data/raw/logP.csv", header=0)
smiles = logP_df['smiles'].tolist()  # Column with SMILES
targets = logP_df['logP'].tolist()  # Column with logP values

logging.info("Data loaded successfully")
logging.info(f"Number of data points: {len(smiles)}")
logging.info(f"Number of targets: {len(targets)}") 

max_length = 200
encoded = tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

#send to device
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
targets = targets.to(device)



# Main script
if __name__ == "__main__":
    # Parameters
    split_size = 0.8
    # dropout = 0.3
    # num_heads = 1
    # num_layers = 3
    # ff_dim = 512
    # batch_size = 32
    num_epochs = 20
    # learning_rate = 1e-4

    train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, targets)

    train_dataset, val_dataset = random_split(train_dataset, [int(split_size * len(train_dataset)), len(train_dataset) - int(split_size * len(train_dataset))])
    train_dataset, test_dataset = random_split(train_dataset, [int(split_size * len(train_dataset)), len(train_dataset) - int(split_size * len(train_dataset))])

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Initalize the combined model and GPU
    # combined_model = LogPPredictionModel(model, dropout=dropout, embed_size=model.config.hidden_size, num_heads=num_heads, ff_hidden_dim=ff_dim, num_layers=num_layers).to(device)
    # Loss and Optimizer
    # criterion = RMSELoss()
    # optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    '''
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, pre_model= model, train_dataset=train_dataset, val_dataset= val_dataset, device=device, epochs= num_epochs), n_trials=30)

    logging.info(f"Best trial: {study.best_trial.value}")

    # Extract the best hyperparameters
    best_params = study.best_trial.params

    logging.info(f"Best hyperparameters: {best_params}")
    # save best hyperparameters to a text file
    with open("../deep_learning_outputs/figures/Hypertuning/STP_v6_best_param_02.txt", "w") as f:
        f.write(f"Best hyperparameters: {best_params}")

    # save all trials to a csv file
    study.trials_dataframe().to_csv("../deep_learning_outputs/figures/Hypertuning/STP_v6_trials_02.csv")
    '''
    # Prepare data loaders with the best batch size
    batch_size = 64 #best_params['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the combined model with the best hyperparameters
    combined_model = LogPPredictionModel(
        model,
        dropout= 0.1, #best_params['dropout'],
        embed_size=model.config.hidden_size,
        num_heads= 1, #best_params['num_heads'],
        ff_hidden_dim= 1024, #best_params['ff_dim'],
        num_layers= 3 #best_params['num_layers']
    ).to(device)

    # Define criterion and optimizer
    criterion = RMSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=10e-8 )#best_params['learning_rate'])


    num_epochs = 300
    # Training Loop
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_STP(combined_model, train_loader, optimizer, criterion, device)
        val_loss,_ ,_ = evaluate_STP(combined_model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")



    
    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig("../deep_learning_outputs/figures/STP_v6_loss_best_param_05.png")
    
    

    # Test the model and plot predictions
    criterion = RMSELoss()
    test_loss, predictions, true_values = evaluate_STP(combined_model, test_loader, criterion, device)

    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.7)
    plt.xlabel("True Values (logP)")
    plt.ylabel("Predicted Values (logP)")
    plt.title("True vs Predicted Values")
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.text(0.5, 0.1, f"RMSE: {criterion(torch.tensor(predictions), torch.tensor(true_values)).item():.4f}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig("../deep_learning_outputs/figures/STP_v6_predict_best_param_05.png")
    
    npz_file = '../deep_learning_outputs/model_evaluation/STP_v6_05.npz'

    np.savez(
        npz_file,
        y_pred=predictions,
        y_test=true_values
    )
    
    # Save the trained model
    torch.save(model.state_dict(), "../deep_learning_outputs/trained_models/STP_v6_best_param_05.pth")

