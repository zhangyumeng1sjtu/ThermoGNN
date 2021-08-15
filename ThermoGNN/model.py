import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, GraphNorm, global_add_pool, global_max_pool, GlobalAttention


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_t, weights=None):
        loss = (y - y_t) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        return torch.mean(loss)


class GNN(nn.Module):

    def __init__(self, num_layer, input_dim, emb_dim, JK = "last" ,drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim),nn.BatchNorm1d(emb_dim),nn.ReLU(),nn.Linear(emb_dim, emb_dim))))
            elif gnn_type == "gcn":
                self.gnns.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_dim, emb_dim))
            else:
                raise ValueError("Invalid GNN type.")


    def forward(self, x, edge_index, edge_attr=None):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)

        return node_representation


class GraphGNN(nn.Module):

    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin") -> object:
        super(GraphGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        
        # self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.out_dim)
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(4*self.emb_dim, self.emb_dim), nn.ReLU(), nn.Dropout(p=self.drop_ratio), nn.Linear(self.emb_dim, self.out_dim))
    
    def forward_once(self, x, edge_index, batch, mut_res_idx):
        node_representation = self.gnn(x, edge_index)

        pooled = self.pool(node_representation, batch)
        mut_node_rep = node_representation[mut_res_idx]

        graph_rep = torch.cat([pooled, mut_node_rep], dim=1)

        return graph_rep

    def forward(self, data):
        out1 = self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch, data.mut_res_idx)
        out2 = self.forward_once(data.x_t, data.edge_index_t, data.x_t_batch, data.mut_res_idx)
        x = torch.cat((out1, out2), dim=-1)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.graph_pred_linear(x)
        return torch.squeeze(x)
