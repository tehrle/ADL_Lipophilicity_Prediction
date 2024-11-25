import torch
from torch.utils.data import random_split

def split_dataset(dataset, ratio=0.80, seed=42):

    # get size of dataset
    dataset_size = len(dataset)

    # get number of entries
    subset_1_size = int(ratio * dataset_size)
    subset_2_size = int(dataset_size - subset_1_size)

    # Split the dataset
    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [subset_1_size, subset_2_size], generator=generator)