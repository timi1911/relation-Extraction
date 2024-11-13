import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    GCN layer with symmetric normalization and two aggregation steps.
    """

    def __init__(self, in_features, hidden_features, out_features):
        super(GraphConvolution, self).__init__()
        # self.weight1 = nn.Parameter(torch.FloatTensor(in_features, hidden_features))
        # self.weight2 = nn.Parameter(torch.FloatTensor(hidden_features, out_features)
        self.fc = nn.Linear(768, 768)
        self.weight1 = nn.Parameter(torch.FloatTensor(in_features, out_features))   #[batch_size, 256, 768]
        nn.init.xavier_uniform_(self.weight1)
        # nn.init.xavier_uniform_(self.weight2)

    def forward(self, input, adj):
        # input: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes] (unnormalized)
        # print("adj.shape", adj.shape)   #[batch_size, 256, 256]
        # print(adj)
        # print("input.shape", input.shape)   #[batch_size, 256, 768]
        # print(input)
        # print("self.weight1", self.weight1.shape)
        # print(self.weight1)
        # First aggregation and transformation



        # x = torch.matmul(input, self.weight1)  # [batch_size, num_nodes, hidden_features]
        #
        # output = torch.tanh(torch.bmm(adj, x))  # [batch_size, num_nodes, out_features]

        #test
        #adj [1, 256, 256]
        #input [1, 256, 768]

        output = self.fc(input)
        # output = output.permute(1, 0, 2)
        # adj = adj.permute(2, 0, 1)
        # output = torch.bmm(adj, output).permute(1, 0, 2)
        output = torch.bmm(adj, output)
        output = F.relu(output)


        return output

    # Example usage

