import torch

import torch.nn as nn

import torch_geometric.nn as pyg_nn


class GraphConvLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(GraphConvLayer, self).__init__()

        self.conv = pyg_nn.GraphConv(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

####feature, adj
    def forward(self, x, edge_index):

        edge_index.squeeze(0)

        x = self.conv(x, edge_index)

        x = self.dropout(x)

        return x