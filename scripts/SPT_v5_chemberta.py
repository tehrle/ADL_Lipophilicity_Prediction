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

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False



from utils import split_SMILE_Dataset, SMILESDataset, TransformerBlock, train_STP_model, evaluate_STP_model,RMSELoss,SMILESAugmentation


class LogPPredictionModel(nn.Module):
    def __init__(self, base_model,dropout):
        super(LogPPredictionModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output)
    





# Initalize the combined model and GPU
combined_model = LogPPredictionModel(model, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model.to(device)

# Initalize Data
logP_df = pd.read_csv("../data/raw/logP.csv", header=0)
smiles = logP_df['smiles'].tolist()  # Column with SMILES
targets = logP_df['logP'].tolist()  # Column with logP values

logging.info("Data loaded successfully")
logging.info(f"Number of data points: {len(smiles)}")
logging.info(f"Number of targets: {len(targets)}") 

max_length = 180
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

# Split the data into training / validation (80/20) and then into training / test (80/20)
train_size = int(0.8 * len(input_ids))
val_size = len(input_ids) - train_size

train_inputs, val_inputs = input_ids[:train_size], input_ids[train_size:]
train_attention_mask, val_attention_mask = attention_mask[:train_size], attention_mask[train_size:]
train_targets, val_targets = targets[:train_size], targets[train_size:]

train_size = int(0.8 * len(train_inputs))
test_size = len(train_inputs) - train_size

train_inputs, test_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_attention_mask, test_attention_mask = train_attention_mask[:train_size], train_attention_mask[train_size:]
train_targets, test_targets = train_targets[:train_size], train_targets[train_size:]

train_dataset = torch.utils.data.TensorDataset(train_inputs, train_attention_mask, train_targets)
val_dataset = torch.utils.data.TensorDataset(val_inputs, val_attention_mask, val_targets)
test_dataset = torch.utils.data.TensorDataset(test_inputs, test_attention_mask, test_targets)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Loss and Optimizer
criterion = RMSELoss()
optimizer = optim.Adam(combined_model.regressor.parameters(), lr=1e-5)




# Main script
if __name__ == "__main__":
    # Parameters
    dropout = 0.1
    num_heads = 8
    num_layers = 10
    ff_dim = 512
    output_dim = 1
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-4
    
    # Training Loop
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        combined_model.train()
        total_loss = 0
        # send trainloader to tqdm for progress bar
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in train_loader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            
            optimizer.zero_grad()
            outputs = combined_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            #outputs = outputs.squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        
        # Evaluate the model on the validation set

        combined_model.eval()
        total_loss = 0

        val_loader = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in val_loader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            with torch.no_grad():
                outputs = combined_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                #outputs = outputs.squeeze()
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        val_losses.append(avg_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")



    
    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig("../deep_learning_outputs/figures/STP_v5_loss_02.png")
    plt.show()
    

    # Test the model and plot predictions
    combined_model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            outputs = combined_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            #outputs = outputs.squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_labels.cpu().numpy())
    true_values = [val for val in true_values]
    predictions = [pred for pred in predictions]
    true_values = np.array(true_values)
    predictions = np.array(predictions)

    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.7)
    plt.xlabel("True Values (logP)")
    plt.ylabel("Predicted Values (logP)")
    plt.title("True vs Predicted Values")
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.text(0.5, 0.1, f"RMSE: {criterion(torch.tensor(predictions), torch.tensor(true_values)).item():.4f}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig("../deep_learning_outputs/figures/STP_v5_predict_02.png")
    plt.show()
    
    
    # Save the trained model
    torch.save(model.state_dict(), "../deep_learning_outputs/trained_models/STP_v5.pth")

