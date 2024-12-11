"""
SMILES Data Preprocessing

This script processes SMILES data from a CSV file and converts it into molecular graphs, molecular descriptors,
molecular images, and extracts various molecular properties.

Authors:
--------
Timo Ehrle

Last Modified:
--------------
10.12.2024
"""
# import necessary packages
import numpy as np
import pandas as pd
import torch
import os
import sys
import logging

# logging settings
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout  # Ensure logging outputs are printed into SLURM output file
)

# settings for hpc
# change wd to current folder to match relative paths
script_dir = os.path.dirname(os.path.abspath(__file__)) # get absolute path of current directory
os.chdir(script_dir)    # change the current directory to the specified directory

# insert top dir to system path for hpc
sys.path.insert(0, '../')
from utils import SMILESConverter
#=======================================================================================================================

#
#   1. Read in Data
#

# Read in data
logP_csv = '../data/raw/logP.csv'
logP_df = pd.read_csv(logP_csv)

logging.info(f'Successfully loaded raw data from {logP_csv}')

# extract SMILES and logP vales as list
smiles_list = logP_df['smiles'].tolist()
logP_list = logP_df['logP'].tolist()

logging.info(f'logP contains {len(logP_list)} SMILES entries')
logging.info('Successfully loaded logP.csv data')

#
#   2. Convert Data
#

logging.info('Starting to process SMILES...')

# Init lists
canonical_smiles_list = []
logP_Crippen_list = []
mol_weights_list = []
number_of_atoms_list = []
mol_images_list = []
mol_graphs_list = []
mol_formula_list = []
mol_descriptors_list = []

# Process smiles
for smiles, logP in zip(smiles_list, logP_list):

    smiles_converter = SMILESConverter(
        smiles=smiles,
        logP=logP,
        image_size=256
    )

    # Convert smiles into RGB image
    mol_images_list.append(np.array(smiles_converter.get_image()))

    # Convert smiles into graphs
    mol_graphs_list.append(smiles_converter.get_graph())

    # Get molecular weight
    mol_weights_list.append(smiles_converter.get_mol_weight())

    # Get number of atoms
    number_of_atoms_list.append(smiles_converter.get_number_of_atoms())

    # Get molecular formula
    mol_formula_list.append(smiles_converter.get_molecular_formula())

    # Get loP value predicted by Crippen's method
    logP_Crippen_list.append(smiles_converter.get_crippen_logP())

    # Get canonial SMILES
    canonical_smiles_list.append(smiles_converter.get_canonical_smiles())

    # Get molecular descriptors
    mol_descriptors_list.append(smiles_converter.get_descriptors())

    # Get descriptor_names
    descriptor_names = smiles_converter.get_descriptor_names()

logging.info(f'Finished processing {len(smiles_list)} SMILES entries')

#
#   3. Save preprocessed SMILES as npz-file
#

logging.info('Saving processed data as npz-file...')

file = '../data/processed/logP_data_v01.npz'

np.savez(
    file=file,
    smiles=np.array(canonical_smiles_list),
    logP=np.array(logP_list),
    logP_Crippen=np.array(logP_Crippen_list),
    mol_images=np.array(mol_images_list),
    mol_weights=np.array(mol_weights_list),
    number_of_atoms=np.array(number_of_atoms_list),
    molecular_formula=np.array(mol_formula_list)
)

logging.info(f'Saved processed data as {file}')

#
#   4. Save PyTorch Graphs as separate file
#

logging.info('Saving PyTorch Graphs...')

file_graphs = '../data/processed/logP_graphs_v01.pt'
torch.save(
    obj=mol_graphs_list,
    f=file_graphs
)

logging.info(f'Saved processed graphs as {file_graphs}')

#
#   5. Save processed information as csv-file
#

logging.info('Saving processed data as csv-file...')

# Create new df
logP_processed_df = pd.DataFrame({
    'smiles':canonical_smiles_list,
    'number_of_atoms':number_of_atoms_list,
    'molecular_formula':mol_formula_list,
    'molecular_weight':mol_weights_list,
    'logP':logP_list,
    'logP_Crippen':logP_Crippen_list
})

csv_file = '../data/processed/logP_processed_v01.csv'

# Save df as csv-file
logP_processed_df.to_csv(csv_file, header=True, index=False)

logging.info(f'Saved processed data as {csv_file}')

#
#   6. Process, clean and save molecular descriptors
#

logging.info('Saving molecular descriptors as csv-file...')

# convert molecular descriptors to df
logP_molecular_descriptors = pd.DataFrame(data=mol_descriptors_list, columns=descriptor_names)

logP_molecular_descriptors.insert(0, 'smiles', canonical_smiles_list)
logP_molecular_descriptors.insert(1, 'logP', logP_list)

# Drop molecular descriptors if they contain NaN values
logP_molecular_descriptors.dropna(axis=1, how='any', inplace=True)

descriptor_csv_file = '../data/processed/logP_molecular_descriptors_v01.csv'

logP_molecular_descriptors.to_csv(descriptor_csv_file, header=True, index=False)

logging.info(f'Saved molecular descriptors data as {descriptor_csv_file}')