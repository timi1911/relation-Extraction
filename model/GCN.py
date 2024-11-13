
import math
from torch import nn
import torch.nn.init as init
import torch
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)


    def forward(self, adjacency, input_feature):
        print("--------------------shape------------------")
        print(input_feature.shape)
        print(self.weight.shape)
        support = torch.einsum('mik,kj->mij', input_feature, self.weight).cuda()
        output = torch.einsum('ki,mij->mkj', adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    # def forward(self, adjacency, input_feature):
    #     support = torch.mm(input_feature, self.weight).cuda()
    #     output = torch.sparse.mm(adjacency, support)
    #     if self.use_bias:
    #         output += self.bias
    #     return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
