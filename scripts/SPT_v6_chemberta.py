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



from utils import RMSELoss


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention layer with residual connection and LayerNorm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward layer with residual connection and LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class LogPPredictionModel(nn.Module):
    def __init__(self, base_model, embed_size, num_heads, ff_hidden_dim, num_layers, dropout):
        super(LogPPredictionModel, self).__init__()
        self.base_model = base_model
        
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Normalization and pooling
        self.norm = nn.LayerNorm(embed_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 1)  # Regression output
        )
    
    def forward(self, input_ids, attention_mask):
        # Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Permute for Transformer processing
        x = hidden_states.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        
        # Forward pass through Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Permute back to (batch_size, seq_len, embed_size)
        x = x.permute(1, 0, 2)
        
        # Apply LayerNorm
        x = self.norm(x)
        
        # Max pooling over the sequence length
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        x = self.max_pool(x).squeeze(-1)  # (batch_size, embed_size)
        
        # MLP Head for regression
        output = self.mlp_head(x)
        return output






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
    # Initalize the combined model and GPU
    combined_model = LogPPredictionModel(model, dropout=dropout, embed_size=model.config.hidden_size, num_heads=num_heads, ff_hidden_dim=ff_dim, num_layers=num_layers).to(device)
    # Loss and Optimizer
    criterion = RMSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-5)
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
    plt.savefig("../deep_learning_outputs/figures/STP_v6_loss_01.png")
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
    plt.savefig("../deep_learning_outputs/figures/STP_v6_predict_01.png")
    plt.show()
    
    
    # Save the trained model
    torch.save(model.state_dict(), "../deep_learning_outputs/trained_models/STP_v6.pth")

