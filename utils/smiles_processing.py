# import necessary packages
from collections import Counter
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Crippen import MolLogP
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
import torch

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