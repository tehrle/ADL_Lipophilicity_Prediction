# import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam
import optuna
import os
import sys
import logging

# logging settings
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout       # Ensure logging outputs are printed into SLURM output file
)

# Redirect stderr to stdout
sys.stderr = sys.stdout

# settings for hpc:
# change wd to current folder to match relative paths
script_dir = os.path.dirname(os.path.abspath(__file__)) # get absolute path of current directory
os.chdir(script_dir)    # change the current directory to the specified directory

# insert top dir to system path for hpc
sys.path.insert(0, '../')
from utils import split_dataset, train_graph_model
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
#   2. Define Model Architecture
#

# register GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f'The following device is used for model training: {device}')

class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_heads=1, dropout=0.0):
        super(SimpleGAT, self).__init__()

        self.dropout = dropout

        # Node and Edge Embeddings
        self.node_embeddings = Linear(in_channels, hidden_channels)
        self.edge_embeddings = Linear(edge_dim, hidden_channels)

        # Graph Attention Layers
        self.conv1 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=self.dropout
        )
        self.gru1 = GRUCell(hidden_channels, hidden_channels)

        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=self.dropout
        )
        self.gru2 = GRUCell(hidden_channels, hidden_channels)

        self.conv3 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=self.dropout
        )
        self.gru3 = GRUCell(hidden_channels, hidden_channels)

        # Global Add Pooling
        self.global_add_pool = global_add_pool

        # Linear Layers for output
        self.linear1 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):

        # Node and Edge Embeddings
        x = F.relu(self.node_embeddings(x))
        edge_attr = F.relu(self.edge_embeddings(edge_attr))

        # GATConv-Layers/ GRU-Layers
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = F.relu(self.gru1(h, x))

        h = F.relu(self.conv2(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = F.relu(self.gru2(h, x))

        h = F.relu(self.conv3(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = F.relu(self.gru3(h, x))

        # Global Add Pooling
        x = self.global_add_pool(x, batch)

        # Output
        x = self.linear1(x)

        return x

#=======================================================================================================================

#
#   3. Tune Dropout rate of model
#

def objective(trial):

    # Set up parameters
    dropout = trial.suggest_float('dropout', 0.0, 0.6)

    # Initialize the model
    model = SimpleGAT(
        in_channels=9,
        edge_dim=3,
        out_channels=1,
        hidden_channels=128,
        num_heads=4,
        dropout=dropout
    )

    # Init optimizer
    optimizer = Adam(model.parameters(), lr=3e-4)

    # Number of epochs for training
    num_epochs = 150

    metrics = train_graph_model(
        model=model,
        num_epochs=num_epochs,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_edge_attr=True
    )

    # extract val_loss after model training
    train_loss = metrics['train_loss'][-1]
    train_mae = metrics['train_mae'][-1]
    train_rmse = metrics['train_rmse'][-1]

    val_loss = metrics['val_loss'][-1]
    val_mae = metrics['val_mae'][-1]
    val_rmse = metrics['val_rmse'][-1]

    # Save metrics to trial
    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('train_mae', train_mae)
    trial.set_user_attr('train_rmse', train_rmse)
    trial.set_user_attr('val_loss', val_loss)
    trial.set_user_attr('val_mae', val_mae)
    trial.set_user_attr('val_rmse', val_rmse)

    return val_loss

logging.info('Starting to tune dropout rate...')

# Create Optuna Study
n_trials = 50
study = optuna.create_study(study_name='basic_gat_v04_dropout_tuning', direction='minimize')
study.optimize(objective, n_trials=n_trials)

logging.info(f'Finished tuning dropout rate, evaluated {n_trials} trials')

# Show best trial
logging.info(f'Best trial: {study.best_trial}, Val_Loss: {study.best_trial.value}')
logging.info(f'Dropout rate of best trial: {study.best_value}')

# Save study as dataframe
study_df = study.trials_dataframe()

study_file = '../deep_learning_outputs/tuning_logs/basic_gat_v04_dropout_tuning.csv'

study_df.to_csv(study_file)

logging.info(f'Saved tuning log to {study_file}')