import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# 文本预处理
def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    return input_ids

# 构建图结构
def build_graph(dependencies,text):
    graph = torch.zeros(len(text), len(text))
    for i, dep in enumerate(dependencies):
        head = dep[0]
        dependent = dep[1]
        print("Head", head, "dependent", dependent)


        graph[dependent][head] = 1
        graph[head][dependent] = 1
    return graph

# GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.linear1(torch.matmul(adj, x)))
        x = self.linear2(torch.matmul(adj, x))
        return x

# CasRel网络
class CasRel(nn.Module):
    def __init__(self, bert_model, gcn_model):
        super(CasRel, self).__init__()
        self.bert_model = bert_model
        self.gcn_model = gcn_model
        self.fc = nn.Linear(2 * bert_model.config.hidden_size, 5)

    def forward(self, input_ids, adj):
        bert_output = self.bert_model(input_ids)[0]
        gcn_output = self.gcn_model(bert_output, adj)
        pooled_output = torch.cat((bert_output[:, 0, :], gcn_output[:, 0, :]), dim=1)
        logits = self.fc(pooled_output)
        return logits

# 示例函数
def example():
    text = "小明的爸爸是中国人"
    dependencies = [(1, 0), (3, 2), (1, 3), (4, 3)]
    num_relations = 23

    input_ids = preprocess_text(text)
    adj = build_graph(dependencies,text)
    print(adj)

    bert_model = BertModel.from_pretrained('bert-base-chinese')
    gcn_model = GCN(bert_model.config.hidden_size, 100, bert_model.config.hidden_size)
    casrel_model = CasRel(bert_model, gcn_model)

    logits = casrel_model(input_ids, adj)
    predicted_relations = torch.argmax(logits, dim=1)

    print("输入文本:", text)
    print("预测关系:", predicted_relations)

example()