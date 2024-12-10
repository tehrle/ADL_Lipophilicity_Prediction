"""
This module provides utilities for handling and splitting datasets for PyTorch workflows.

Functions:
-----------
- split_dataset: Splits a PyTorch dataset into two subsets based on a given ratio.
- split_SMILE_Dataset: Splits a SMILES dataset into two subsets.
- split_SMILES: Splits raw SMILES data and associated targets into two subsets.

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024
"""

# import necessary packages
import torch
from torch.utils.data import random_split
from .smiles_processing import SMILESDataset
#=======================================================================================================================

def split_dataset(dataset, ratio=0.80, seed=42):
    """
    This function splits a PyTorch dataset into two subsets based on a specified ratio.

    Parameters:
    -----------
    dataset : torch_geometric.data.Dataset
        The dataset to be split.
    ratio : float, optional
        The fraction of the first subset's size. Defaults to 0.80.
    seed : int, optional
        The random seed for reproducibility. Defaults to 42.

    Returns:
    --------
    tuple of torch.utils.data.Subset
        Two subsets of the dataset split according to the specified ratio.
    """

    # get size of dataset
    dataset_size = len(dataset)

    # get number of entries
    subset_1_size = int(ratio * dataset_size)
    subset_2_size = int(dataset_size - subset_1_size)

    # Split the dataset
    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [subset_1_size, subset_2_size], generator=generator)

def split_SMILE_Dataset(dataset, ratio, seed=None):
    '''
    This function splits the dataset's SMILES strings, targets, and tokenizer
    into two subsets based on the specified ratio.

    Parameters:
    -----------
    dataset : SMILESDataset
        The dataset to be split.
    ratio : float
        The ratio of the first subset's size to the total dataset size.
    seed : int, optional
        The random seed for reproducibility. Defaults to None.

    Returns:
    --------
    tuple of SMILESDataset
        Two SMILESDataset instances representing the split subsets.
    '''
    if seed is not None:
        torch.manual_seed(seed)
    
    total_size = len(dataset)
    split_size = int(total_size * ratio)
    remaining_size = total_size - split_size

    smiles = dataset.get_smiles()
    targets = dataset.get_targets()
    tokenizer = dataset.get_tokenizer()

    # Split the dataset into two parts with random indices based on the ratio
    split1_smiles = []
    split1_targets = []
    split2_smiles = []
    split2_targets = []

    indices = torch.randperm(total_size)
    split1_indices = indices[:split_size]
    split2_indices = indices[split_size:]

    for idx in split1_indices:
        split1_smiles.append(smiles[idx])
        split1_targets.append(targets[idx])
    
    for idx in split2_indices:
        split2_smiles.append(smiles[idx])
        split2_targets.append(targets[idx])

    # Create new instances of the original dataset class
    split1_dataset = SMILESDataset(split1_smiles, split1_targets, tokenizer)
    split2_dataset = SMILESDataset(split2_smiles, split2_targets, tokenizer)
    
    return split1_dataset, split2_dataset

def split_SMILES(smiles, targets, ratio, seed=None):
    """
    This function splits lists of SMILES strings and their corresponding targets into two subsets based on the
    specified ratio.

    Parameters:
    -----------
    smiles : list
        List of SMILES strings.
    targets : list
        List of target values corresponding to the SMILES strings.
    ratio : float
        The ratio of the first subset's size to the total dataset size.
    seed : int, optional
        The random seed for reproducibility. Defaults to None.

    Returns:
    --------
    tuple
        Four lists: SMILES strings and targets for the first subset, and SMILES strings
        and targets for the second subset.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    total_size = len(smiles)
    split_size = int(total_size * ratio)
    
    # Split the dataset into two parts with random indices based on the ratio
    split1_smiles = []
    split1_targets = []
    split2_smiles = []
    split2_targets = []

    indices = torch.randperm(total_size)
    split1_indices = indices[:split_size]
    split2_indices = indices[split_size:]

    for idx in split1_indices:
        split1_smiles.append(smiles[idx])
        split1_targets.append(targets[idx])
    
    for idx in split2_indices:
        split2_smiles.append(smiles[idx])
        split2_targets.append(targets[idx])

    return split1_smiles, split1_targets, split2_smiles, split2_targets

