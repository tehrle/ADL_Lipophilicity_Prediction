import pandas as pd
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

from utils import (
    RMSELoss
)

# Load the ChemBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Load your CSV file
logP_df = pd.read_csv("../data/raw/logP.csv", header=0)
smiles_list = logP_df['smiles'].tolist()  # Column with SMILES
targets_list = logP_df['logP'].tolist()  # Column with logP values

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

# Define the custom dataset
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets_list, tokenizer, max_length=180):
        self.smiles_list = smiles_list
        self.targets_list = targets_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets_list[idx]

        # Tokenize SMILES
        encoded = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Extract input_ids and attention_mask
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return input_ids, attention_mask, torch.tensor(target, dtype=torch.float32)

# Main script
if __name__ == "__main__":
    # Parameters
    max_len = 180  # Maximum SMILES length
    dropout = 0.1
    num_heads = 8
    num_layers = 4
    ff_dim = 512
    output_dim = 1
    batch_size = 32  # Start with a smaller batch size
    num_epochs = 200
    learning_rate = 1e-4
    validation_split = 0.2  # 20% of data for validation

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prediction_layer = nn.Linear(model.config.hidden_size, 1)
    prediction_layer.to(device)
    optimizer = optim.Adam(prediction_layer.parameters(), lr=learning_rate)
    loss_fn = RMSELoss()

    # Initialize dataset
    full_dataset = SMILESDataset(smiles_list, targets_list, tokenizer, max_length=max_len)

    # Calculate lengths for train and validation splits
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        # Training Phase
        model.eval()  # Model remains in eval mode since parameters are frozen
        prediction_layer.train()  # Set prediction layer to train mode

        epoch_train_loss = 0.0
        # Wrap the training DataLoader with tqdm
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
    

        for batch in train_iterator:
            input_ids, attention_mask, batch_targets = batch

            # Move batch tensors to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                molecule_embeddings = embeddings.mean(dim=1)

            predictions = prediction_layer(molecule_embeddings)
            loss = loss_fn(predictions.squeeze(), batch_targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        prediction_layer.eval()  # Set prediction layer to eval mode

        epoch_val_loss = 0.0

        # Wrap the validation DataLoader with tqdm
        val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
    
        with torch.no_grad():
            for batch in val_iterator:
                input_ids, attention_mask, batch_targets = batch

                # Move batch tensors to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                molecule_embeddings = embeddings.mean(dim=1)

                predictions = prediction_layer(molecule_embeddings)
                loss = loss_fn(predictions.squeeze(), batch_targets)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig("../deep_learning_outputs/figures/SPT_v4_loss_01.png", dpi=300)


    # Evaluate on validation set for plotting true vs predicted values
    model.eval()
    prediction_layer.eval()
    all_true_values = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, batch_targets = batch

            # Move batch tensors to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            molecule_embeddings = embeddings.mean(dim=1)

            predictions = prediction_layer(molecule_embeddings)

            all_true_values.extend(batch_targets.cpu().numpy())
            all_predictions.extend(predictions.squeeze().cpu().numpy())

    plt.figure()
    plt.scatter(all_true_values, all_predictions, alpha=0.7)
    plt.xlabel("True Values (logP)")
    plt.ylabel("Predicted Values (logP)")
    plt.title("True vs Predicted Values (Validation Set)")
    plt.plot([min(all_true_values), max(all_true_values)],
             [min(all_true_values), max(all_true_values)], 'r--')
    plt.savefig("../deep_learning_outputs/figures/SPT_v4_loss_01.png", dpi=300)

    # Save the trained prediction layer
    torch.save({
        'model_state_dict': model.state_dict(),
        'prediction_layer_state_dict': prediction_layer.state_dict()
    }, "../deep_learning_outputs/trained_models/smiles_property_predictor.pth")
