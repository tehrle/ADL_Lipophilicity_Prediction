# import necessary packages
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Crippen import MolLogP
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch_geometric.data import Data
import torch

# TODO: Add Docstrings for class

class SMILESConverter:
    def __init__(self, smiles, image_size=128, atom_featurizer=CanonicalAtomFeaturizer(),
                 bond_featurizer=CanonicalBondFeaturizer()):

        self.smiles = smiles
        self.image_size = image_size
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

        self.mol = self.smiles_to_mol()
        self.graph = self.mol_to_graph()

    def smiles_to_mol(self):

        mol = MolFromSmiles(self.smiles)   # convert smile into mol object

        # TODO: Check if we want to explicitly add Hydrogen atoms to mol-object to obtain them as nodes in graph (otherwise they will be "ignored")

        # check if conversion was successful
        if mol is None:
            return None

        return mol

    def get_mol_weight(self):
        return MolWt(self.mol)

    def get_crippen_logP(self):
        return MolLogP(self.mol)

    def get_number_of_atoms(self):
        return self.mol.GetNumAtoms()

    def get_image(self):
        return MolToImage(self.mol, size=(self.image_size, self.image_size))

    def mol_to_graph(self):
        # Create DGL graph
        dgl_graph = mol_to_bigraph(
            mol=self.mol,
            node_featurizer=self.atom_featurizer,  # Atom features from DGL-Life-Sciences
            edge_featurizer=self.bond_featurizer  # Bond features from DGL-Life-Sciences
        )

        # Extract node features from DGL graph
        node_features = dgl_graph.ndata['h']  # Node features: Shape [num_nodes, num_node_features]

        # Extract edge features from DGL graph
        edge_features = dgl_graph.edata['e']  # Edge features: Shape [num_edges, num_edge_features]

        # Extract edge index (source and destination nodes of edges)
        src, dst = dgl_graph.edges()  # Returns two tensors: source nodes and destination nodes
        edge_index = torch.stack([src, dst], dim=0)  # Shape [2, num_edges]

        # Create PyTorch Geometric Data object
        graph = Data(
            x=node_features,  # Node features
            edge_index=edge_index,  # Edge indices
            edge_attr=edge_features  # Edge features
        )

        return graph

    def get_graph(self):
        return self.graph







