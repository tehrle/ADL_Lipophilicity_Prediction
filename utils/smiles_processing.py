# import necessary packages
from collections import Counter
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Crippen import MolLogP
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset


from pysmilesutils.augment import SMILESAugmenter

# TODO: Add Docstrings for class

class SMILESConverter:
    def __init__(self, smiles, logP, image_size=128):

        self.smiles = smiles
        self.logP = logP
        self.image_size = image_size

        self.mol = self.smiles_to_mol()
        self.graph = self.mol_to_graph()

    def smiles_to_mol(self):

        mol = MolFromSmiles(self.smiles)   # convert smile into mol object

        # TODO: Check if we want to explicitly add Hydrogen atoms to mol-object to obtain them as nodes in graph
        #  (otherwise they will not appear as nodes, however they are added as node features anyway).

        # check if conversion was successful
        if mol is None:
            return None

        return mol

    def get_mol_weight(self):
        return MolWt(self.mol)

    def get_crippen_logP(self):
        return MolLogP(self.mol)

    def get_number_of_atoms(self):

        # Add hydrogens to mol-object to get the right count
        mol_h = AddHs(self.mol)

        return mol_h.GetNumAtoms()

    def get_molecular_formula(self):

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
        return MolToImage(self.mol, size=(self.image_size, self.image_size))

    def mol_to_graph(self):

        # Create PyG Graph
        graph = from_smiles(self.smiles)

        # Convert Node mat into float
        graph.x = graph.x.float()

        # Convert edge attributes into float
        graph.edge_attr = graph.edge_attr.float()

        # Add logP value to graph
        graph.y = torch.tensor(self.logP, dtype=torch.float).view(1, -1)

        return graph

    def get_graph(self):
        return self.graph
    
    

    

class SMILESDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_length=180):
        """
        Initializes the SMILES processing class.
        Args:
            smiles (list): The list of SMILES strings.
            targets (list): The list of target values associated with the SMILES string.
            tokenizer (object): The tokenizer object used to tokenize the SMILES string.
        """
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        """
        Returns the SMILES string and the target value at the given index.
        Args:
            idx (int): The index of the SMILES string and target value.
        Returns:
            tuple: The SMILES string and the target value.
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
        return self.smiles
    
    def get_targets(self):
        return self.targets
    
    def get_tokenizer(self):
        return self.tokenizer
        



    
class SMILESAugmentation:
    def __init__(self, train_dataset, tokenizer):
        """
        Initializes the SMILES processing class with a training dataset and a tokenizer.
        Args:
            train_dataset (Dataset): The dataset containing SMILES strings and their corresponding targets.
            tokenizer (Tokenizer): The tokenizer used to decode SMILES strings.
        Attributes:
            train_dataset (Dataset): Stores the provided training dataset.
            tokenizer (Tokenizer): Stores the provided tokenizer.
        Iterates over the training dataset, decodes each SMILES string using the tokenizer, and processes the data.
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
            

        
        
    
    def augment(self, num_samples):
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

        
        