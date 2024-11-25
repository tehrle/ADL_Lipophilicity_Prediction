# import necessary packages
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch_geometric.nn import AttentiveFP
import os
import sys
import logging

# logging settings
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# settings for hpc:
# change wd to current folder to match relative paths
script_dir = os.path.dirname(os.path.abspath(__file__)) # get absolute path of current directory
os.chdir(script_dir)    # change the current directory to the specified directory

# insert top dir to system path for hpc
sys.path.insert(0, '../')
from utils import split_dataset, train_graph_model, plot_history, predict_graph_model, RMSELoss, plot_logp_correlation

#=======================================================================================================================

#
#   1. Load and preprocess data / graphs
#

logging.info('Starting to load and process graph...')

mol_graphs_list = torch.load('../data/processed/logP_graphs_v01.pt')  # list of torch_geometric.Data graphs

logging.info(f'Loaded {len(mol_graphs_list)} graphs')

logging.info(f'Example Graph for Structure: {mol_graphs_list[0]}')

train_data, test_data = split_dataset(mol_graphs_list, ratio=0.8, seed=42)  # split data into train and test
train_data, val_data = split_dataset(train_data, ratio=0.8, seed=42)        # split train_data into train and validation set

logging.info(
    f'Splited data set into the following sizes: '
    f'Train:{len(train_data)},'
    f'Validation:{len(val_data)},'
    f'Test:{len(test_data)}'
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

#=======================================================================================================================

#
#   2. Load and setup model for training
#

logging.info('Setting up model for training...')

# register GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f'The following device is used for model training: {device}')

model = AttentiveFP(
    in_channels=74,
    hidden_channels=128,
    out_channels=1,
    edge_dim=12,
    num_layers=4,
    num_timesteps=4,
    dropout=0.2
)

# model summary
# need to figure out how to print model summary using torch_geometric.nn.summary



#=======================================================================================================================

#
#   3. Train Model
#

logging.info('Starting to train model...')

# Init optimizer
optimizer = Adam(model.parameters(), lr=3e-4)

# Number of epochs for training
num_epochs = 100

metrics = train_graph_model(
    model=model,
    num_epochs=num_epochs,
    optimizer=optimizer,
    loss_fn=nn.MSELoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)

logging.info(f'Finished: Trained model for {num_epochs} epochs')

# save trained model
torch.save(model, '../deep_learning_outputs/trained_models/attentiveFP_v01.pt')

logging.info('Successfully saved trained model')

# plot trainings history
plot_history(
    train_history=metrics['train_loss'],
    val_history=metrics['val_loss'],
    train_label='Train MSE (Loss) / -',
    val_label='Val MSE (Loss) / -',
    figsize=(16, 8),
    file='../deep_learning_outputs/figures/train_history_attentiveFP_v01.png'
)

# save train/val_losses as csv
training_lod_df = pd.DataFrame(metrics)

# save trainings log
training_lod_df.to_csv(
    '../deep_learning_outputs/training_logs/train_history_attentiveFp_v01.csv',
    index=False,
    header=True
)

logging.info('Successfully saved training logs')

#=======================================================================================================================

#
#   4. Evaluate Model
#

logging.info('Evaluating model on test set...')

# Make predictions on test set
y_pred_tensor, y_test_tensor = predict_graph_model(
    model=model,
    test_loader=test_loader,
    device=device
)

logging.info(f'Shape of y_pred_tensor: {y_pred_tensor.shape}')
logging.info(f'Shape of y_test_tensor: {y_test_tensor.shape}')

# Initialize Metrics
mse = nn.MSELoss()
mae = nn.L1Loss()
rmse = RMSELoss()

# calculate metrics for evaluation
mse_loss = mse(y_pred_tensor, y_test_tensor)
mae_loss = mae(y_pred_tensor, y_test_tensor)
rmse_loss = rmse(y_pred_tensor, y_test_tensor)

# Get Metrics
logging.info(f'Mean Squared Error: {mse_loss.item()}')
logging.info(f'Mean Absolute Error: {mae_loss.item()}')
logging.info(f'Root Mean Squared Error: {rmse_loss.item()}')

# Convert Metrics to np.ndarray for plotting
y_pred_array = y_pred_tensor.flatten().numpy()
y_test_array = y_test_tensor.flatten().numpy()

logging.info(f'Shape of y_pred_array: {y_pred_array.shape}')
logging.info(f'Shape of y_test_array: {y_test_array.shape}')

# Plot predicted vs. True logP Values
correlation_file = f'../deep_learning_outputs/figures/attentiveFP_correlation_plot_v01.png'

plot_logp_correlation(
    file=correlation_file,
    y_true=y_test_array,
    y_pred=y_pred_array,
    xlim=(-6, 18),
    ylim=(-6, 18)
)

logging.info(f'Successfully saved correlation plot as {correlation_file}')