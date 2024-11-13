from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
# 加载预训练的BERT模型和分词器
model_name = 'bert-base-chinese'  # 使用中文预训练的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入句子
sentence = "我爱吃苹果并且我喜欢看电影"

# # 对句子进行分词
#
# tokens = tokenizer.tokenize(sentence)
#
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
#
# # 创建attention mask，用于指示哪些token是真实的，哪些是填充的
#
# attention_mask = [1] * len(token_ids)

# # 将token IDs和attention mask转换为Tensor
#
# print(token_ids)
# print(len(token_ids))        #13
# print(len(attention_mask))   #13
#
# print(type(token_ids))       #<class 'list'>
# print(type(attention_mask))  #<class 'list'>
#
#
#
# token_ids = torch.tensor([token_ids])
#
# attention_mask = torch.tensor([attention_mask])
#
# # 构建输入字典
#
# inputs = {'input_ids': token_ids, 'attention_mask': attention_mask}
#
# # 将输入传递给BERT模型
#
# outputs = model(**inputs)
#
# # 提取每个单词的编码向量
#
# word_vectors = outputs[0]
#
# for i, word in enumerate(tokens):
#     print(f"词: {word}")
#
#     print(f"向量: {word_vectors[0][i].shape}")
#
#     print()




tokenized = tokenizer(sentence, max_length=256, truncation=True)
tokens = tokenized['input_ids']
masks = tokenized['attention_mask']



# print(tokens)
# print(len(tokens))
# print(len(masks))

# print(type(tokens))
# print(type(masks))

tokens = torch.tensor([tokens])

masks = torch.tensor([masks])

inputs = {'input_ids': tokens, 'attention_mask': masks}

# print(type(inputs))

outputs = model(**inputs)
word_vectors = outputs[0]

# print(word_vectors.shape)
# print(type(word_vectors))

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def dataPreprocessForGCN(text):
    nlp = StanfordCoreNLP(r'F:\CUG\代码\stanford-corenlp-full-2018-10-05\stanford-corenlp-full-2018-10-05', lang='zh')

    sentence = text
    # 依存句法分析
    dependency = nlp.dependency_parse(sentence)
    # #分词
    words = list(nlp.word_tokenize(sentence))  # words   :   ['蔡志坚', '在', '南京', '艺术', '学院', '求学', '时', '受过', '系统', '、', '正规',

    dependency.sort(key=lambda x: x[2])  # dependency   :   [('nsubj', 8, 1), ('case', 5, 2), ('compound:nn', 5, 3), ('compound:nn', 5, 4),........]

    # embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=maxlen)
    # tmp = [idx for idx, w in enumerate(words)]
    # tmp = torch.LongTensor(tmp)
    #
    # embed = embedding(tmp)

    idx1 = [arc[1] for arc in dependency]
    idx2 = [arc[2] for arc in dependency]
    idx = []
    for i in range(len(idx1)):
        idx.append([idx1[i], idx2[i]])
        idx.append([idx2[i], idx1[i]])

    edge_index = torch.tensor(idx, dtype=torch.long)

    # data = Data(x=embed, edge_index=edge_index.t().contiguous())

    nlp.close()


    # 12.15添加 将依存关系转换为adj矩阵
    adj = np.zeros((256,256))
    for i,j in edge_index:
        adj[i-1][j-1] = 1
    #######

    #edge_index -> adj, data -> input_feature
    # return edge_index,words
    return adj,words


adj, words = dataPreprocessForGCN(sentence)

print("adj--------->", adj.shape)
print("adj--------->", type(adj))

print("word_vectors---------------->", word_vectors.shape)
print("word_vectors shape[0]-------->", word_vectors.shape[0])
word_vectors = word_vectors.squeeze(0)
print("word_vectors---------------->", word_vectors.shape)

input_feature = []

for word in words:
    head_id = find_head_idx(sentence, word)

    target_embedding = torch.mean(word_vectors[head_id: head_id + len(word)], dim=0)

    print(target_embedding.shape)

    input_feature.append(target_embedding)

#########扩容256个节点的特征
zero = torch.zeros(768)
for i in range(256 - len(input_feature)):
    input_feature.append(zero)

adj = torch.as_tensor(adj, dtype=torch.int)
input_feature = torch.stack(input_feature)

# for i, word in enumerate(tokens):
#     print(f"词: {word}")
#
#     print(f"向量: {word_vectors[0][i].shape}")
#
#     print()