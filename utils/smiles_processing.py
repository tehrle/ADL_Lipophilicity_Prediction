"""
This module provides utilities for processing and augmenting SMILES strings.

Classes:
--------
- SMILESConverter: Converts and processes SMILES strings into molecular representations and graph structures.
- SMILESDataset: Handles datasets containing SMILES strings and their associated target values.
- SMILESAugmentation: Augments SMILES data for data augmentation purposes.

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024
"""

# import necessary packages
from collections import Counter
from rdkit.Chem import MolFromSmiles, MolToSmiles, AddHs, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Crippen import MolLogP
from torch_geometric.utils import from_smiles
import torch
from torch.utils.data import Dataset
from pysmilesutils.augment import SMILESAugmenter
#=======================================================================================================================

class SMILESConverter:
    """
    Converts and processes SMILES strings into molecular representations, descriptors, images, and graph structures.

    Attributes:
    -----------
    smiles : str
        The input SMILES string.
    logP : float
        The logP value associated with the molecule.
    image_size : int
        The size of the generated molecule image.
    mol : rdkit.Chem.Mol
        The RDKit molecule object derived from the SMILES string.
    mol_h : rdkit.Chem.Mol
        The RDKit molecule object with added hydrogens.
    canonical_smiles : str
        The canonicalized SMILES string using rdkit.
    graph : torch_geometric.data.Data
        The graph representation of the molecule for PyTorch Geometric.
    calc : rdkit.ML.Descriptors.MoleculeDescriptors.MolecularDescriptorCalculator
        Calculator for molecular descriptors.

    Methods:
    --------
    smiles_to_mol():
        Converts the SMILES string to an RDKit molecule object.
    mol_to_smiles():
        Converts the molecule to a canonical SMILES string.
    get_mol_weight():
        Computes the molecular weight of the molecule.
    get_crippen_logP():
        Computes the Crippen logP value of the molecule.
    get_number_of_atoms():
        Returns the number of atoms in the molecule (including hydrogens).
    get_molecular_formula():
        Generates the molecular formula of the molecule.
    get_image():
        Returns an image of the molecule.
    mol_to_graph():
        Converts the molecule to a PyTorch Geometric graph representation.
    get_graph():
        Returns the graph representation of the molecule.
    get_canonical_smiles():
        Returns the canonical SMILES string.
    get_descriptors():
        Returns molecular descriptors for the molecule.
    get_descriptor_names():
        Returns the names of available molecular descriptors.
    """
    def __init__(self, smiles, logP, image_size=128):
        """
        Initializes the SMILESConverter with a SMILES string, logP value, and image size.

        Parameters:
        -----------
        smiles : str
            The input SMILES string.
        logP : float
            The logP value associated with the molecule.
        image_size : int, optional
            The size of the generated molecule image (in pixels). Defaults to 128.
        """
        self.smiles = smiles
        self.logP = logP
        self.image_size = image_size

        self.mol = self.smiles_to_mol()
        self.mol_h = AddHs(self.mol)
        self.canonical_smiles = self.mol_to_smiles()
        self.graph = self.mol_to_graph()
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

    def smiles_to_mol(self):
        """
        Converts the SMILES string to an RDKit molecule object.

        Returns:
        --------
        rdkit.Chem.Mol
            The RDKit molecule object. Returns None if conversion fails.
        """
        mol = MolFromSmiles(self.smiles)   # convert smile into mol object

        # check if conversion was successful
        if mol is None:
            return None

        return mol

    def mol_to_smiles(self):
        """
        Converts the molecule to a canonical SMILES string.

        Returns:
        --------
        str
            The canonical SMILES string.
        """
        return MolToSmiles(self.mol)

    def get_mol_weight(self):
        """
        Computes the molecular weight of the molecule.

        Returns:
        --------
        float
            The molecular weight of the molecule.
        """
        return MolWt(self.mol)

    def get_crippen_logP(self):
        """
        Computes the Crippen logP value of the molecule.

        Returns:
        --------
        float
            The Crippen logP value of the molecule.
        """
        return MolLogP(self.mol)

    def get_number_of_atoms(self):
        """
        Returns the number of atoms in the molecule, including hydrogens.

        Returns:
        --------
        int
            The number of atoms in the molecule.
        """
        return self.mol_h.GetNumAtoms()

    def get_molecular_formula(self):
        """
        Generates the molecular formula of the molecule.

        Returns:
        --------
        str
            The molecular formula.
        """

        atoms_counts = Counter()

        # Count number of atoms
        for atom in self.mol.GetAtoms():

            atom_symbol = atom.GetSymbol()
            atoms_counts[atom_symbol] += 1

            # add Hydrogens
            if atom.GetTotalNumHs() > 0:
                atoms_counts['H'] += atom.GetTotalNumHs() # only add H's to counter if present

        # Sort molecular formula; 1. Carbon, 2. Hydrogen, other atoms in alphabetic order
        sorted_atoms = []

        if 'C' in atoms_counts:
            sorted_atoms.append(('C', atoms_counts['C']))

        if 'H' in atoms_counts:
            sorted_atoms.append(('H', atoms_counts['H']))

        for atom in sorted(atoms_counts.keys()):
            if atom not in ('C', 'H'):
                sorted_atoms.append((atom, atoms_counts[atom]))

        # Convert list into string
        molecular_formula = []
        for atom, count in sorted_atoms:
            if count > 1:
                molecular_formula.append(f'{atom}{count}')

            else:
                molecular_formula.append(atom)

        return ''.join(molecular_formula)

    def get_image(self):
        """
        Returns an RGB image of the molecule.

        Returns:
        --------
        PIL.Image.Image
            An image of the molecule with the specified image size.
        """
        return MolToImage(self.mol, size=(self.image_size, self.image_size))

    def mol_to_graph(self):
        """
        Converts the molecule to a PyTorch Geometric graph representation.

        Returns:
        --------
        torch_geometric.data.Data
            The graph representation of the molecule.
        """

        # Create PyG Graph
        graph = from_smiles(self.canonical_smiles)

        # Convert Node mat into float
        graph.x = graph.x.float()

        # Convert edge attributes into float
        graph.edge_attr = graph.edge_attr.float()

        # Add logP value to graph
        graph.y = torch.tensor(self.logP, dtype=torch.float).view(1, -1)

        return graph

    def get_graph(self):
        """
        Returns the precomputed graph representation of the molecule.

        Returns:
        --------
        torch_geometric.data.Data
            The graph representation of the molecule.
        """
        return self.graph

    def get_canonical_smiles(self):
        """
        Returns the canonical SMILES string of the molecule.

        Returns:
        --------
        str
            The canonical SMILES string.
        """
        return self.canonical_smiles

    def get_descriptors(self):
        """
         Computes molecular descriptors for the molecule.

         Returns:
         --------
         list of float
             A list of molecular descriptor values.
         """
        return self.calc.CalcDescriptors(self.mol_h)

    def get_descriptor_names(self):
        """
        Returns the names of the available molecular descriptors.

        Returns:
        --------
        list of str
            A list of molecular descriptor names.
        """
        return self.calc.GetDescriptorNames()

class SMILESDataset(Dataset):
    """
    A dataset class for SMILES strings and their associated target values.

    This class extends the PyTorch `Dataset` class and provides utilities for
    handling SMILES data, including tokenization and padding.

    Attributes:
    -----------
    smiles : list
        List of SMILES strings.
    targets : list
        List of target values associated with the SMILES strings.
    tokenizer : object
        Tokenizer used for encoding SMILES strings.
    max_length : int
        Maximum length for SMILES encoding. Default is 180.

    Methods:
    --------
    __len__():
        Returns the size of the dataset.
    __getitem__(idx):
        Returns the encoded SMILES string and target value at the given index.
    get_smiles():
        Returns the list of SMILES strings.
    get_targets():
        Returns the list of target values.
    get_tokenizer():
        Returns the tokenizer object.
    """
    def __init__(self, smiles, targets, tokenizer, max_length=180):
        """
        Initializes the SMILESDataset with SMILES strings, target values, a tokenizer, and a maximum encoding length.

        Parameters:
        -----------
        smiles : list
            List of SMILES strings.
        targets : list
            List of target values associated with the SMILES strings.
        tokenizer : object
            Tokenizer used for encoding SMILES strings.
        max_length : int, optional
            Maximum length for SMILES encoding. Default is 180.
        """
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
        --------
        int
            The number of SMILES strings in the dataset.
        """
        return len(self.smiles)
    
    def __getitem__(self, idx):
        """
        Returns the encoded SMILES string and the target value at the specified index.

        The SMILES string is tokenized and padded to the specified maximum length.

        Parameters:
        -----------
        idx : int
            The index of the SMILES string and target value to retrieve.

        Returns:
        --------
        tuple
            A tuple containing:
            - encoded_smiles (torch.Tensor): The padded and tokenized SMILES string.
            - target (torch.Tensor): The target value corresponding to the SMILES string.
        """
        smiles = self.smiles[idx]
        target = self.targets[idx]
        encoded_smiles = self.tokenizer.encode(smiles)[0]
        # pad the encoded_smiles tensor to max_length
        if len(encoded_smiles) < self.max_length:
            pad_length = self.max_length - len(encoded_smiles)
            encoded_smiles = torch.cat([encoded_smiles, torch.zeros(pad_length, dtype=torch.long)])
        else:
            encoded_smiles = encoded_smiles[:self.max_length]
        target = torch.tensor(target, dtype=torch.float)
        return encoded_smiles, target
    
    def get_smiles(self):
        """
        Returns the list of SMILES strings in the dataset.

        Returns:
        --------
        list
            The list of SMILES strings.
        """
        return self.smiles
    
    def get_targets(self):
        """
        Returns the list of target values associated with the SMILES strings.

        Returns:
        --------
        list
            The list of target values.
        """
        return self.targets
    
    def get_tokenizer(self):
        """
        Returns the tokenizer used for encoding SMILES strings.

        Returns:
        --------
        object
            The tokenizer object.
        """
        return self.tokenizer

class SMILESAugmentation:
    """
    Augments SMILES strings in a training dataset for data augmentation.

    This class facilitates the generation of additional SMILES strings from an
    existing training dataset using random augmentations. It supports datasets
    in various formats, including PyTorch datasets and lists.

    Attributes:
    -----------
    train_dataset : Dataset or list
        The training dataset containing SMILES strings and their targets.
    tokenizer : object, optional
        Tokenizer used to decode SMILES strings. Required for certain dataset types.
    smiles : list
        List of SMILES strings extracted from the training dataset.
    targets : list
        List of target values corresponding to the SMILES strings.

    Methods:
    --------
    augment(num_samples):
        Generates augmented SMILES strings by applying random augmentations.
    """
    def __init__(self, train_dataset, tokenizer = None):
        """
        Initializes the SMILESAugmentation class with a training dataset and an optional tokenizer.

        Parameters:
        -----------
        train_dataset : Dataset or list
            The training dataset containing SMILES strings and their targets.
        tokenizer : object, optional
            Tokenizer used to decode SMILES strings.

        Attributes:
        -----------
        train_dataset : Dataset or list
            Stores the provided training dataset.
        tokenizer : object or None
            Stores the provided tokenizer.
        smiles : list
            List of SMILES strings extracted from the training dataset.
        targets : list
            List of target values corresponding to the SMILES strings.
        """
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer

        # check if train_dataset is an instance of SMILESDataset or a subset of pytorch Dataset
        if isinstance(train_dataset, SMILESDataset):
            self.smiles = train_dataset.get_smiles()
            self.targets = train_dataset.get_targets()
            
        elif isinstance(train_dataset, torch.utils.data.Subset):
            self.smiles = []
            self.targets = []
            for i in train_dataset:
                self.smiles.append(self.tokenizer.decode(i[0]))
                self.targets.append(i[1])

        elif isinstance(train_dataset, list):
            self.smiles = train_dataset[0]
            self.targets = train_dataset[1]

    def augment(self, num_samples):
        """
        Generates augmented SMILES strings and their corresponding targets.

        This method uses a SMILESAugmenter to create augmented SMILES strings
        by applying random augmentations to the original dataset.

        Parameters:
        -----------
        num_samples : int
            The number of augmented SMILES strings to generate per original SMILES string.

        Returns:
        --------
        tuple of lists
            - augmented_smiles (list): The augmented SMILES strings.
            - augmented_targets (list): The corresponding target values.
        """
        augmenter = SMILESAugmenter()
        augmented_smiles = []
        augmented_targets = []

        for smiles, target in zip(self.smiles, self.targets):
            augmented_smiles.append(smiles)
            augmented_targets.append(target)
            
            
            new_smiles = augmenter([smiles]*num_samples)
            for i in range(num_samples):
                augmented_smiles.append(new_smiles[i])
                augmented_targets.append(target)

        # shuffle the augmented data
        indices = len(augmented_smiles)
        indices = torch.randperm(indices)
        augmented_smiles = [augmented_smiles[i] for i in indices]
        augmented_targets = [augmented_targets[i] for i in indices]

        return augmented_smiles, augmented_targets