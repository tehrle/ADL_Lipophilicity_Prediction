# import necessary packages
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
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
from utils import plot_logp_correlation
#=======================================================================================================================

#
#   1. Load and preprocess data / graphs
#

logging.info('Started to load and process data set containing molecular descriptors')

# load data set
descriptors_df = pd.read_csv('../data/processed/logP_molecular_descriptors_v01.csv')

# split data into features and labels
y = descriptors_df['logP'].to_numpy()
X = descriptors_df.drop(['logP', 'smiles'], axis=1).to_numpy()

logging.info(f'Loaded descriptors shape: {X.shape}')
logging.info(f'Labels shape: {y.shape}')

# split data into train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logging.info(
    'Splited data set into the following sizes:'
    f'Train: {len(y_train)},'
    f'Test: {len(y_test)},'
)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logging.info('Finished preprocessing data')
#=======================================================================================================================

#
#   2. Setup model and tune hyperparameters
#

logging.info('Tuning Hyperparameters for XGBRegressor')

# tune hyperparameters
trial_results = []

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    }

    # Train and evaluate model
    model = xgb.XGBRegressor(**param)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

    # Compute mean and std of negative MSE
    mean_score = -scores.mean()
    std_score = scores.std()

    # Save the trial's parameters and scores
    trial_results.append({
        'n_estimators': param['n_estimators'],
        'max_depth': param['max_depth'],
        'learning_rate': param['learning_rate'],
        'mean_mse': mean_score,
        'std_mse': std_score
    })

    return mean_score

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)

# Save all trial results as CSV
results_csv = '../deep_learning_outputs/training_logs/xgb_regressor_hyperparameter_tuning_v01.csv'

# Convert trial results to DataFrame
trial_results_df = pd.DataFrame(trial_results)

# Save to CSV
trial_results_df.to_csv(
    results_csv,
    index=False,
    header=True
)

logging.info(f'Successfully saved hyperparameter tuning results as {results_csv}')

# Best parameters
logging.info(f'Finished hyperparameter tuning with best parameters: {study.best_params}')
#=======================================================================================================================

#
#   3. Train Model using optimized hyperparameters
#

# extract best hyperparameters
n_estimators = study.best_params['n_estimators']
max_depth = study.best_params['max_depth']
learning_rate = study.best_params['learning_rate']

logging.info(f'Training model using the following hyperparameters:'
             f'n_estimators: {n_estimators}, '
             f'max_depth: {max_depth}, '
             f'learning_rate: {learning_rate}')

# initialize and train model
model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

model.fit(X_train_scaled, y_train)

logging.info('Finished training model')

# get training scores
mse_training = mean_squared_error(y_train, model.predict(X_train_scaled))
mae_training = mean_absolute_error(y_train, model.predict(X_train_scaled))
rmse_training = root_mean_squared_error(y_train, model.predict(X_train_scaled))

logging.info(f'MSE training: {mse_training}')
logging.info(f'MAE training: {mae_training}')
logging.info(f'RMSE training: {rmse_training}')

#=======================================================================================================================

#
#   4. Evaluate Model
#

logging.info('Evaluating model on test set...')

# make prediction
y_pred = model.predict(X_test_scaled)

mse_test = mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = root_mean_squared_error(y_test, y_pred)

logging.info(f'MSE Test: {mse_test}')
logging.info(f'MAE Test: {mae_test}')
logging.info(f'RMSE Test: {rmse_test}')

# Combine parameters with train/test metrics
summary_csv = '../deep_learning_outputs/model_evaluation/xgb_regressor_summary_v01.csv'

summary_df = pd.DataFrame({
    'n_estimators': [n_estimators],
    'max_depth': [max_depth],
    'learning_rate': [learning_rate],
    'train_mse': [mse_training],
    'train_mae': [mae_training],
    'train_rmse': [rmse_training],
    'test_mse': [mse_test],
    'test_mae': [mae_test],
    'test_rmse': [rmse_test]
})

# Save to CSV
summary_df.to_csv(
    summary_csv,
    index=False,
    header=True
)

logging.info(f'Successfully saved summary of hyperparameter tuning and evaluation as {summary_csv}')

# Save predicted values
npz_file = '../deep_learning_outputs/model_evaluation/xgb_regressor_predicted_values_v01.npz'

np.savez(
    npz_file,
    y_pred=y_pred,
    y_test=y_test
)

logging.info(f'Successfully saved labels/ predicted values of test set as {npz_file}')

# Plot predicted vs. True logP Values
correlation_file = f'../deep_learning_outputs/model_evaluation/xgb_regressor_correlation_plot_v01.png'

plot_logp_correlation(
    file=correlation_file,
    y_true=y_test,
    y_pred=y_pred,
    xlim=(-6, 18),
    ylim=(-6, 18)
)

logging.info(f'Successfully saved correlation plot as {correlation_file}')