import torch
from torch.utils.data import random_split
from .smiles_processing import SMILESDataset

def split_dataset(dataset, ratio=0.80, seed=42):

    # get size of dataset
    dataset_size = len(dataset)

    # get number of entries
    subset_1_size = int(ratio * dataset_size)
    subset_2_size = int(dataset_size - subset_1_size)

    # Split the dataset
    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [subset_1_size, subset_2_size], generator=generator)



def split_SMILE_Dataset(dataset, ratio, seed=None):
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


    
