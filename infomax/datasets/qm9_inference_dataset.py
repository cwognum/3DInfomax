import os
from typing import List

import torch
import dgl
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm


class QM9InferenceDataset(Dataset):

    def __init__(self, smiles_list: List[str], reprocess: bool = False, device='cuda:0'):

        self.directory = 'dataset/QM9Inference'
        self.processed_file = 'qm9inference_processed.pt'
        self.device = device

        self.failed_indices = []

        # load the data and get normalization values
        if not os.path.exists(os.path.join(self.directory, 'processed', self.processed_file)) or reprocess:
            self.process(smiles_list)
        data_dict = torch.load(os.path.join(self.directory, 'processed', self.processed_file))

        self.features_tensor = data_dict['atom_features']
        self.e_features_tensor = data_dict['edge_features']
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {k: data_dict[k] for k in ('mol_id', 'edge_slices', 'atom_slices', 'n_atoms')}

        self.dgl_graphs = {}
        self.avg_degree = data_dict['avg_degree']

    def __len__(self):
        return len(self.meta_dict['mol_id'])

    def __getitem__(self, idx):
        e_start = self.meta_dict['edge_slices'][idx].item()
        e_end = self.meta_dict['edge_slices'][idx + 1].item()
        start = self.meta_dict['atom_slices'][idx].item()
        n_atoms = self.meta_dict['n_atoms'][idx].item()
        return self.get_graph(idx, e_start, e_end, start, n_atoms)

    def get_graph(self, idx, e_start, e_end, start, n_atoms):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx].to(self.device)
        else:
            edge_indices = self.edge_indices[:, e_start: e_end]
            g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
            g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
            g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
            self.dgl_graphs[idx] = g.to("cpu")
            return g

    def process(self, smiles_list: List[str]):

        print(f'Processing {len(smiles_list)} SMILES strings and saving processed dataset to '
              f'{os.path.abspath(os.path.join(self.directory, "processed"))}')

        mol_ids = []
        atom_counts = []
        atom_slices = [0]
        edge_slices = [0]
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        avg_degree = 0  # average degree in the dataset
        atomic_numbers = []

        self.failed_indices = []

        for mol_idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list)):

            try:
                # get the molecule using the smiles representation from the csv file
                mol = Chem.MolFromSmiles(smi)
                # add hydrogen bonds to molecule because they are not in the smiles representation
                mol = Chem.AddHs(mol)

                n_atoms = len(mol.GetAtoms())

                atom_features_list = []
                atomic_numbers_list = []
                for atom in mol.GetAtoms():
                    atom_features_list.append(atom_to_feature_vector(atom))
                    atomic_numbers_list.append(atom.GetAtomicNum())

                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)

                # Graph connectivity in COO format with shape [2, num_edges]
                edge_index = torch.tensor(edges_list, dtype=torch.long).T
                edge_features = torch.tensor(edge_features_list, dtype=torch.long)
                avg_degree += (len(edges_list) / 2) / n_atoms
                total_edges += len(edges_list)
                total_atoms += n_atoms

                # Only if no exception was thrown, save the data
                mol_ids.append(mol_idx)
                edge_slices.append(total_edges)
                atom_slices.append(total_atoms)
                atom_counts.append(n_atoms)
                edge_indices.append(edge_index)
                all_edge_features.append(edge_features)
                all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))
                atomic_numbers.extend(atomic_numbers_list)

            except Exception as error:
                self.failed_indices.append(mol_idx)
                print(f"Failed to process {smi} due to {error}.")

        data_dict = {
            'mol_id': torch.tensor(mol_ids, dtype=torch.long),
            'n_atoms': torch.tensor(atom_counts, dtype=torch.long),
            'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
            'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
            'edge_indices': torch.cat(edge_indices, dim=1),
            'atom_features': torch.cat(all_atom_features, dim=0),
            'edge_features': torch.cat(all_edge_features, dim=0),
            'atomic_number_long': torch.tensor(atomic_numbers, dtype=torch.long)[:, None],
            'avg_degree': avg_degree / len(mol_ids)
        }

        os.makedirs(os.path.join(self.directory, 'processed'), exist_ok=True)
        torch.save(data_dict, os.path.join(self.directory, 'processed', self.processed_file))

        print(f"Failed to process a total of {len(self.failed_indices)} SMILES")
        for mol_idx in self.failed_indices:
            smi = smiles_list[mol_idx]
            print(f"({mol_idx}) - {smi if smi != '' else '<empty>'}")