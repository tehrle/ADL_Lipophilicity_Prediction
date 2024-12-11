import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from pysmilesutils.tokenize import SMILESAtomTokenizer
import sys
import os
import logging
import time as t

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    smiles, targets = zip(*batch)
    smiles = [torch.tensor(s, dtype=torch.long) for s in smiles]  # Convert each SMILES sequence to a tensor
    targets = torch.tensor(targets, dtype=torch.float)
    return smiles, targets


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

from utils import (split_dataset,split_SMILE_Dataset, SMILESDataset, 
                    TransformerBlock, train_STP_model, evaluate_STP_model,
                    RMSELoss,SMILESAugmentation, STP)






# Training and Evaluation


# Main script
if __name__ == "__main__":
    # Load your CSV file
    logP_df = pd.read_csv("../data/raw/logP.csv", header=0)
    smiles = logP_df['smiles'].tolist()  # Column with SMILES
    targets = logP_df['logP'].tolist()  # Column with logP values

    # Parameters
    max_len = 180  # Maximum SMILES length
    embed_dim = 128
    dropout = 0.1
    num_heads = 8
    num_layers = 4
    ff_dim = 512
    output_dim = 1
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-4

    # Tokenizer and Dataset
    # tokenizer = SMILESTokenizer()
    # tokenizer.fit(smiles)
    tokenizer = SMILESAtomTokenizer(smiles=smiles)


    # Split the dataset into training and validation
    dataset = SMILESDataset(smiles, targets, tokenizer, max_length=max_len)
    train_dataset, val_dataset = split_SMILE_Dataset(dataset, ratio=0.8, seed=42)
    train_dataset, test_dataset = split_SMILE_Dataset(train_dataset, ratio=0.8, seed=42)
    

    # Traindata augmentation
    augmentation = SMILESAugmentation(train_dataset, tokenizer)
    smiles, targets = augmentation.augment(num_samples=2)
    train_dataset = SMILESDataset(smiles, targets, tokenizer, max_length=max_len)


    # logging shape infos
    logging.info(f'Train data shape: {len(train_dataset)}')
    logging.info(f'Validation data shape: {len(val_dataset)}')
    logging.info(f'Test data shape: {len(test_dataset)}')

    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    model = STP(len(tokenizer), embed_dim, num_heads, ff_dim, num_layers, dropout, max_len).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #loss_fn = nn.MSELoss()
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
    plt.show()

    # Test the model and plot predictions
    _, predictions, true_values = evaluate_STP_model(model, val_loader, loss_fn, device)
    predictions = [pred[0] for pred in predictions]

    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.7)
    plt.xlabel("True Values (logP)")
    plt.ylabel("Predicted Values (logP)")
    plt.title("True vs Predicted Values")
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), "../deep_learning_output/trained_models/smiles_property_predictor.pth")
