import torch
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()

        # 假设我们使用一个GAT卷积层
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        # 如果你需要多个GAT层，你可以继续添加如下：
        # self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=num_heads, concat=False, dropout=0.6)

        # 如果你在最后一层不使用concat（即concat=False），则输出通道数应与hidden_channels相匹配
        self.lin = torch.nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index):
        # GAT卷积层
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.6, training=self.training)

        # 如果有多个GAT层，继续传递x
        # x = self.conv2(x, edge_index)

        # 线性层，用于调整输出通道数
        x = self.lin(x)

        return x

    # 示例用法


# num_nodes = 1000  # 图中节点的数量
# in_channels = 16  # 输入特征维度
# hidden_channels = 8  # 隐藏层维度
# out_channels = 16  # 输出特征维度
# num_heads = 8  # GAT中的头数（注意力机制的数量）
#
# # 假设我们已经有了一些节点特征和边索引
# node_features = torch.randn((num_nodes, in_channels))
# edge_index = torch.randint(0, num_nodes, (2, 5000), dtype=torch.long)  # 随机生成边索引作为示例
#
# model = GAT(in_channels, hidden_channels, out_channels, num_heads)
# output = model(node_features, edge_index)