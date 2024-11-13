import json
from random import choice

import tokenizers
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
import torch
import numpy as np
from transformers import BertTokenizer,BertModel
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

#GCN数据构建
from stanfordcorenlp import StanfordCoreNLP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_rel = 18
maxlen = 256

def load_data(train_path, dev_path, test_path, rel_dict_path):
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
    loader = JsonLoader({"text": "text", "spo_list": "spo_list"})
    data_bundle = loader.load(paths)
    id2rel = json.load(open(rel_dict_path,'rb'))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    return data_bundle, rel_vocab


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1



# # GCN Label处理
# def make_entity_label(source, label_list ,target):
#     target_len = len(target)
#     for i in range(len(source)):
#         if source[i: i + target_len] == target:
#             for j in range(target_len):
#                 label_list[i + j] = 1
#     return label_list
# #GCN Label处理
# def find_word_label(source, label_list ,target):
#     target_len = len(target)
#     for i in range(len(source)):
#         if source[i: i + target_len] == target:
#             for j in range(target_len):
#                 if label_list[i + j] != 1:
#                     return 0
#             return 1
#     return 0


nlp = StanfordCoreNLP(r'D:\workplace\jxg\stanford-corenlp-full-2018-10-05\stanford-corenlp-full-2018-10-05', lang='zh')

def dataPreprocessForGCN(text):

    sentence = text

    if len(sentence) >= 256:
        sentence = sentence[:254]


    # 依存句法分析
    dependency = nlp.dependency_parse(sentence)
    # #分词
    words = list(nlp.word_tokenize(sentence))  # words   :   ['蔡志坚', '在', '南京', '艺术', '学院', '求学', '时', '受过', '系统', '、', '正规',
    # nlp.close()
    dependency.sort(key=lambda x: x[2])  # dependency   :   [('nsubj', 8, 1), ('case', 5, 2), ('compound:nn', 5, 3), ('compound:nn', 5, 4),........]

    # embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=maxlen)
    # tmp = [idx for idx, w in enumerate(words)]
    # tmp = torch.LongTensor(tmp)
    #
    # embed = embedding(tmp)
    word_len = []
    word_len.append(0)
    for i in words:
        word_len.append(len(i))

    for i in range(len(word_len)):
        if i != 0:
            word_len[i] = word_len[i-1] + word_len[i]

    idx1 = [arc[1] for arc in dependency]
    idx2 = [arc[2] for arc in dependency]
    idx = [[], []]


    for i in range(len(idx1)):
        if idx1[i] == 0 or idx2[i] == 0:
            continue

        # 头指向头，每个分词的第一个字符位置
        idx[0].append(word_len[idx2[i] - 1])
        idx[1].append(word_len[idx1[i]])

        #尾指向尾，每个分词的最后一个字符位置
        idx[0].append(word_len[idx2[i]] - 1)
        idx[1].append(word_len[idx1[i]] - 1)

        # # 头指向尾，每个分词的第一个字符指向最后一个字符
        # idx[0].append(word_len[idx2[i] - 1])
        # idx[1].append(word_len[idx2[i]] - 1)
        # idx[0].append(word_len[idx1[i] - 1])
        # idx[1].append(word_len[idx1[i]] - 1)

        # idx[0].append(idx2[i])
        # idx[1].append(idx1[i])

    edge_index = torch.tensor(idx, dtype=torch.long)

    # data = Data(x=embed, edge_index=edge_index.t().contiguous())

######注意，这里理解错误，图卷积网络模型需要的edge_index输入格式，是[2,N]，表示连接关系，可以直接用依存关系套入。
    # 12.15添加 将依存关系转换为adj矩阵
    # adj = np.zeros((256, 256))
    # for i, j in edge_index:
    #     adj[i-1][j-1] = 1
    #######




    # 创建一个256x256的全零矩阵作为邻接矩阵
    adjacency_matrix = np.zeros((256, 256), dtype=np.float)

    # 从边数组中获取边的数量
    num_edges = edge_index.shape[1]

    # 遍历每一条边，并在邻接矩阵中标记边的存在
    for i in range(num_edges):
        source, target = edge_index[:, i]
        # 确保节点编号在有效范围内
        assert 0 <= source < 256, f"Source node number {source} out of range."
        assert 0 <= target < 256, f"Target node number {target} out of range."
        # 在邻接矩阵中将对应位置设置为1，表示存在一条从source到target的边
        adjacency_matrix[source, target] = 1
        adjacency_matrix[target, source] = 1

    np.fill_diagonal(adjacency_matrix, 1)


    # 计算节点的度（这里假设是无权图，使用二值邻接矩阵）
    node_degrees = np.sum(adjacency_matrix, axis=1)

    # 为了防止除以零，我们添加一个小的常数值到节点度中
    epsilon = 1e-5
    node_degrees = np.where(node_degrees == 0, epsilon, node_degrees)
    # print(node_degrees)

    # 计算度矩阵的逆平方根
    degree_inv_sqrt = np.power(node_degrees, -0.5)
    degree_inv_sqrt_matrix = np.diag(degree_inv_sqrt)

    # 对邻接矩阵进行归一化
    adj_normalized = degree_inv_sqrt_matrix @ adjacency_matrix @ degree_inv_sqrt_matrix

    # 现在 adj_normalized 是归一化后的邻接矩阵
    # print(adj_normalized)


    #
    # #########################################   Adj数据处理
    # epsilon = 1e-4
    # degrees = adjacency_matrix.sum()  # [num_nodes, 1]
    # degrees = torch.clamp(degrees, min=epsilon)  # 确保 degrees 中没有零值
    # # Symmetric normalization
    # degrees_inv_sqrt = torch.pow(degrees + epsilon, -0.5)
    # # degrees_inv_sqrt = degrees.pow(-0.5)  # [num_nodes, 1]
    #
    # print("degrees_inv_sqrt", degrees_inv_sqrt)
    # print(degrees_inv_sqrt.shape)
    # adj_normalized = torch.mm(torch.mm(degrees_inv_sqrt, adjacency_matrix), degrees_inv_sqrt)
    #
    # # degrees_inv_sqrt_expanded = degrees_inv_sqrt.expand_as(adjacency_matrix)  # [batch_size, num_nodes, num_nodes]
    # # norm_adj = torch.matmul(degrees_inv_sqrt_expanded, adjacency_matrix)  # [batch_size, num_nodes, num_nodes]
    # # norm_adj = torch.matmul(norm_adj, degrees_inv_sqrt_expanded.transpose(-2, -1))  # [batch_size, num_nodes, num_nodes]

    # print("data_process________ADJ_____Normalized")

        # 打印邻接矩阵（可选）
    # print(adj_normalized.shape)

    # print(adj_normalized)

    # print(adj)

    #edge_index -> adj, data -> input_feature
    # nlp.close()
    return adj_normalized, words
    # return adj,words
    # return idx, words

bert_path = 'D:/workplace/jxg/relationExtraction/bert-base-chinese'

class MyDataset(DataSet):
    def __init__(self, config, dataset, rel_vocab, is_test):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.is_test = is_test
        # self.BertModel = BertModel.from_pretrained('bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.BertModel = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)


    def __getitem__(self, item):
        json_data = self.dataset[item]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.config.max_len, truncation=True)
        tokens = tokenized['input_ids']
        masks = tokenized['attention_mask']
        # print("masks:",len(masks))



###############################
        #获取GCN所需数据，邻接矩阵和分词特征编码
        #adj -> [token_size,2]
        #input_feature -> [token_size,maxlen]      maxlen -> 256

        #12.16改动，获取adj邻接矩阵  ——  256*256
        # adj, words = dataPreprocessForGCN(text)
        adj, words = dataPreprocessForGCN(text)

        #input 存的是 bert的vocab对照的token_id
        input_feature = []
        gcn_masks = []

########用BERT模型获取每一个字符的向量，那么每一个分词的向量即为字符向量和的平均值

        input_ids_gcn = torch.tensor([tokens])

        attention_mask = torch.tensor([masks])

        inputs = {'input_ids': input_ids_gcn, 'attention_mask': attention_mask}

        # print(type(inputs))

        outputs = self.BertModel(**inputs)
        word_vectors = outputs[0].squeeze(0)



        for word in words:
            head_id = find_head_idx(text, word)
            target_embedding = torch.mean(word_vectors[head_id: head_id + len(word)], dim=0)


            # print(target_embedding.shape)

            input_feature.append(target_embedding)

#########扩容256个节点的特征
        zero = torch.zeros(768)
        for i in range(256 - len(input_feature)):
            input_feature.append(zero)


        # outputs = self.BertModel(words)
        # word_vector = outputs[0]


        # #将依存句法树的每个节点依次做tokenizer，token_ids存入input_feature -> [token_ids.len, token_id.len]
        # for i in range(len(words)):
        #     # gcn_tokenized = self.tokenizer(words[i], max_length=self.config.max_len, truncation=True)
        #
        #     gcn_tokenized = self.tokenizer(words[i], max_length=self.config.max_len, truncation=True)
        #
        #
        #     input_feature.append(gcn_tokenized['input_ids'])
        #     gcn_masks.append(gcn_tokenized['attention_mask'])
##############################

        text_len = maxlen
        # text_len = len(tokens)
        # print("text_len",text_len)
        # print(type(tokens))
        tokens = self.sequence_padding(tokens, maxlen=maxlen)
        masks = self.sequence_padding(masks, maxlen=maxlen)


        #二维矩阵words_token_id做padding
        # input_feature = self.sequence_padding_2d(input_feature, maxlen=maxlen)      #[maxlen, eachToekn_id_len] -> [256, xxx]
        # gcn_masks = self.sequence_padding_2d(gcn_masks, maxlen=maxlen)



        token_ids = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool)


#############
        adj = torch.as_tensor(adj, dtype=torch.float)
        input_feature = torch.stack(input_feature)
        gcn_masks = torch.tensor(gcn_masks, dtype=torch.bool)
##############



        sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
        sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
        obj_heads = torch.zeros((text_len, self.config.num_relations)) #num_relations = 18
        obj_tails = torch.zeros((text_len, self.config.num_relations))


#######################
        GCN_label = []
        label_list = []
        label_list = self.sequence_padding(label_list, maxlen=maxlen)
#######################




        #以下代码为搭建对应数据的指针网络，实体起始、终止位置下标置为1
        if not self.is_test:
            s2ro_map = defaultdict(list)
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])

                #以下为GCN label处理——label
                # label_list = make_entity_label(text, label_list, spo['subject'])
                # label_list = make_entity_label(text, label_list, spo['object'])
                #以上为GCN Label处理





                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

# #############################
#         # #这里需要设计GCN图节点的label，根据指针网络设置
#         # gcn_sub_label = []
#         # gcn_obj_label = []
#         # find_word_label
#         for i in range(len(words)):
#             GCN_label.append(find_word_label(text, label_list, words[i]))
#         GCN_label = self.sequence_padding(GCN_label, maxlen=maxlen)
#         GCN_label = torch.tensor(GCN_label, dtype=torch.long)
# ##############################





        #tokens存储的是vocab中的id，token_ids是对tokens转换成了Tensor
        #masks里的值是转换成了bool类型，也是Tensor
        # return token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, json_data['spo_list']
        return token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, json_data['spo_list'], adj, input_feature

    def __len__(self):
        return len(self.dataset)

    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding]*(maxlen-len(x))]) if len(x)<maxlen else np.array(x[:maxlen])
        return list(output)


##########################

    def sequence_padding_2d(self, x, maxlen, padding=0):
        output = x
        if len(x) < maxlen:
            for i in range(maxlen - len(x)):
                output.append([padding])
        else:
            output = np.array(x[:maxlen])

        output = [self.sequence_padding(i, 256, 0) for i in output]

        return output
###########################

def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, adj, input_feature = zip(*batch)
    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True)

#####################
    batch_adj = pad_sequence(adj, batch_first=True)
    batch_input_feature = pad_sequence(input_feature, batch_first=True)

    # batch_GCN_label = pad_sequence(GCN_label, batch_first=True)
    # batch_gcn_masks = pad_sequence(gcn_masks, batch_first=True)
###########################





    return {"token_ids": batch_token_ids.to(device),
            "mask": batch_masks.to(device),
            "sub_head": batch_sub_head.to(device),
            "sub_tail": batch_sub_tail.to(device),
            "sub_heads": batch_sub_heads.to(device),

            "adj": batch_adj.to(device),
            "input_feature": batch_input_feature.to(device),
            # "GCN_label": batch_GCN_label.to(device),
            # "gcn_masks": batch_gcn_masks.to(device)

            }, \
           {"mask": batch_masks.to(device),
            "sub_heads": batch_sub_heads.to(device),
            "sub_tails": batch_sub_tails.to(device),
            "obj_heads": batch_obj_heads.to(device),
            "obj_tails": batch_obj_tails.to(device),
            "triples": triples,

            "adj": batch_adj.to(device),
            "input_feature": batch_input_feature.to(device),
            # "GCN_label": batch_GCN_label.to(device),
            # "gcn_masks": batch_gcn_masks.to(device)
            }


class MyRandomSampler(Sampler):
    def __call__(self, data_set):
        return np.random.permutation(len(data_set)).tolist()


def get_data_iterator(config, dataset, rel_vocab, is_test=False, collate_fn=my_collate_fn):
    dataset = MyDataset(config, dataset, rel_vocab, is_test)
    return TorchLoaderIter(dataset=dataset,
                           collate_fn=collate_fn,
                           batch_size=config.batch_size if not is_test else 1,
                           sampler=MyRandomSampler())



