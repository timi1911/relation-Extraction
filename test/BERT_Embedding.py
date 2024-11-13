import torch
from transformers import BertTokenizer, BertModel

import transformers
print(transformers.__file__)

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 输入句子
sentence = "我喜欢自然语言处理"
# 将句子转换为BERT模型的输入格式

tokenized = tokenizer(sentence, max_length=256, truncation=True)
tokens = tokenized['input_ids']
masks = tokenized['attention_mask']


#####################
# 获取字符的索引
input_ids_char = [tokenizer.encode(c) for c in sentence]
input_ids = torch.tensor(input_ids_char).unsqueeze(0)

# 获取每个字符的向量
embeddings = model.embeddings(input_ids)

# 输出每个字符的向量
for token, embedding in zip(sentence, embeddings.squeeze(0)):
    print(f"Token: {token}, Embedding: {embedding}")

#
#
# # print(embeddings.shape)