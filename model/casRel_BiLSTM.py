import torch.nn as nn
import torch
from transformers import BertModel
#以下为修改部分
from axial_attention import AxialAttention
from einops import rearrange


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        #这里可以添加相关模型，BiLSTM、axialAttention

        #BiLSTM，提取上下文信息，输入768维度，输出768//2，层数为1
        self.word_lstm = nn.LSTM(
        input_size=768,
        hidden_size=768 // 2,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=True)

        # #轴向注意力机制
        # self.axialatt = AxialAttention(
        #              dim = 768,               # embedding dimension
        #              dim_index = 1,         # where is the embedding dimension
        #              dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
        #              heads = 1,             # number of heads for multi-head attention
        #              num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        #              sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
        #             )



    #Bert编码
    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    #主实体识别，Linear+sigmoid
    def get_subs(self, encoded_text):

        #修改尝试，添加BiLSTM
        lstm_output, (h,c) = self.word_lstm(encoded_text)
        # print("lstm_output.shape",lstm_output.shape)
        # print("lstm_output.shape",lstm_output.shape) # lstm_output.shape   torch.Size([8, 256, 768])
        # 8,256,768

        # #修改尝试，添加AxialAttention
        # output = torch.reshape(lstm_output,(-1,16,16,768))
        # output = output.permute(0,3,1,2)
        # # 8, 768,16,16
        # output = self.axialatt(output)
        # print("output.shape",output.shape)  #output.shape torch.Size([8, 768, 16, 16])

        # b1, c1, h1 , w1 = output.shape
        # output = rearrange(output, 'b1 c1 h1 w1 -> b1 c1 (h1 w1)')
        # output = output.permute(0,2,1)
        # print("output.shape 2 -> ",output.shape)

        # b1, c1, h1 , w1 = output.shape
        # output = rearrange(output, 'b1 c1 h1 w1 -> b1 c1 (h1 w1)')
        # output = output.permute(0,2,1)
        # 8,256,768
        #output -> 8,256,768
        #input_feature -> 8,token_size,256
        # GCN_input = torch.matmul(output, gcn_feature)

        # surport = self.conv1(adj, gcn_feature)
        # gcn_output = self.conv2(adj, surport)
        # print("gcn_output.shape", gcn_output.shape)

        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(lstm_output))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(lstm_output))
        return pred_sub_heads, pred_sub_tails

    #客实体识别，Linear+sigmoid
    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # sub_head_mapping[batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub

        #做Linear + sigmoid运算，计算客实体的头尾位置
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, token_ids, mask, sub_head, sub_tail):
        #这一步，进行bert编码
        encoded_text = self.get_encoded_text(token_ids, mask)
        #get_subs，对编码层进行一个线性层+一个Sigmoid激活函数进行判断，判断每个token是不是头实体的开始或结束
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)


        #以下为客实体和关系预测模块

        #pytorch中的unsqueeze函数，改变向量维度，参数是几，就是再第几维度上加1，如向量shape=[3,4],unsqueeze(1) -> [3,1,4]
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)

        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }