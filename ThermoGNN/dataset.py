import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from ThermoGNN.utils.weights import assign_weights


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


def load_dataset(graph_dir, split="train", labeled=True):

    data_list = []
    num_nodes = 0
    num_edges = 0

    for i, name in enumerate(open(f"data/{split}_names.txt")):
        name = name.strip()
        G_wt = nx.read_gpickle(f"{graph_dir}/{split}/{name}_wt.pkl")
        data_wt = from_networkx(G_wt)
        G_mut = nx.read_gpickle(f"{graph_dir}/{split}/{name}_mut.pkl")
        data_mut = from_networkx(G_mut)

        data_direct = PairData(data_wt.edge_index, data_wt.x,
                               data_mut.edge_index, data_mut.x)


        data_direct.mut_res_idx = G_wt.graph['mut_pos']

        data_reverse = PairData(data_mut.edge_index, data_mut.x,
                                data_wt.edge_index, data_wt.x)
        data_reverse.mut_res_idx = G_mut.graph['mut_pos']

        if labeled:
            data_direct.y = G_wt.graph['y']
            data_reverse.y = -G_mut.graph['y']

        data_list.append(data_direct)
        data_list.append(data_reverse)

        if split == "train":
            weights = assign_weights("data/datasets/curated_data_direct.rmdup.avg.csv")
            data_direct.wy = torch.tensor(weights[i])
            data_reverse.wy = torch.tensor(weights[i])

        num_nodes += data_wt.num_nodes
        num_nodes += data_mut.num_nodes
        num_edges += data_wt.num_edges
        num_edges += data_mut.num_edges

    print(f'{split.upper()} DATASET:')
    print(f'Number of nodes: {num_nodes / len(data_list):.2f}')
    print(f'Number of edges: {num_edges / len(data_list):.2f}')
    print(f'Average node degree: {num_edges / num_nodes:.2f}')

    return data_list
