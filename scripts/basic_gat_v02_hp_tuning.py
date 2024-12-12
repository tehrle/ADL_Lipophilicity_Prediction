# import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_add_pool, summary
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
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

# Extract logP-values for scaling
train_targets = torch.cat([d.y for d in train_data], dim=0).numpy()
val_targets = torch.cat([d.y for d in val_data], dim=0).numpy()

#Init scaler and scale samples
scaler = MinMaxScaler()
train_targets_scaled = scaler.fit_transform(train_targets)
val_targets_scaled = scaler.transform(val_targets)

# overwrite unscaled logP values in graphs
for i in range(len(train_data)):
    train_data[i].y = torch.tensor(train_targets_scaled[i], dtype=torch.float).unsqueeze(-1)

for i in range(len(val_data)):
    val_data[i].y = torch.tensor(val_targets_scaled[i], dtype=torch.float).unsqueeze(-1)
#=======================================================================================================================

#
#   2. Load and setup model for training
#

class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_heads=1, negative_slope=0.01, dropout=0.0):
        super(SimpleGAT, self).__init__()

        self.negative_slope = negative_slope
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
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=self.dropout
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=hidden_channels,
            heads=num_heads,
            concat=False,
            dropout=self.dropout
        )

        # Global Add Pooling
        self.global_add_pool = global_add_pool

        # Linear Layers for output
        self.linear1 = Linear(hidden_channels, hidden_channels // 2)
        self.linear2 = Linear(hidden_channels // 2, hidden_channels // 4)
        self.linear3 = Linear(hidden_channels // 4, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):

        # Node and Edge Embeddings
        x = F.leaky_relu(self.node_embeddings(x), negative_slope=self.negative_slope)
        edge_attr = F.leaky_relu(self.edge_embeddings(edge_attr), negative_slope=self.negative_slope)

        # GATConv-Layers
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr), negative_slope=self.negative_slope)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr), negative_slope=self.negative_slope)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr), negative_slope=self.negative_slope)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Add Pooling
        x = self.global_add_pool(x, batch)

        # Output
        x = F.leaky_relu(self.linear1(x), negative_slope=self.negative_slope)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.linear2(x), negative_slope=self.negative_slope)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear3(x)

        return x

model = SimpleGAT(
    in_channels=9,
    hidden_channels=128,
    out_channels=1,
    edge_dim=3,
    num_heads=4,
    negative_slope=0.01
)
#=======================================================================================================================

#
#   3. Tune hyperparameters of model
#

# register GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f'The following device is used for model training: {device}')

def objective(trial):

    # Set up parameters
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
    num_heads = trial.suggest_int('num_heads', 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = SimpleGAT(
        in_channels=9,
        edge_dim=3,
        out_channels=1,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout
    )

    # Init optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

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
n_trials = 100
study = optuna.create_study(study_name='basic_gat_hp_tuning_v01', direction='minimize')
study.optimize(objective, n_trials=n_trials)

logging.info(f'Finished tuning basic_gat_v02 model, evaluated {n_trials} trials')

# Show best trial
logging.info(f'Best trial: {study.best_trial}, Val_Loss: {study.best_trial.value}')
logging.info(f"Best trial's parameters: {study.best_trial.params}")

# Save study as dataframe
study_df = study.trials_dataframe()

study_file = '../deep_learning_outputs/tuning_logs/basic_gat_hp_tuning_v01.csv'

study_df.to_csv(study_file)

logging.info(f'Saved tuning log to {study_file}')