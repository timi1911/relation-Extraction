from stanfordcorenlp import StanfordCoreNLP
import torch
import numpy as np
# 注意：这里的路径应该是到stanford-corenlp文件夹的父文件夹的路径
# 并且确保已经下载了中文模型并设置了正确的properties文件
nlp = StanfordCoreNLP(r'F:\CUG\代码\stanford-corenlp-full-2018-10-05', lang='zh')

# 中文句子
sentence = "我喜欢学习自然语言处理。"

# 进行依存句法分析
# 注意：这里的方法名可能需要根据你的库版本进行调整
dependency_parse = nlp.parse(sentence)

# 打印依存句法分析结果
# 注意：这里的结果结构也可能需要根据实际的库输出进行调整
for dep in dependency_parse:
    # 假设dep是一个包含'governor', 'dependent', 'dep'等字段的字典
    print(
        f"{dep['governor']}-{dep.get('governorGloss', '')} <- {dep['dependent']}-{dep.get('dependentGloss', '')} [{dep['dep']}]")

# 关闭Stanford CoreNLP服务
# 注意：这个库可能不需要显式地关闭，但如果有必要，你可以调用相应的关闭方法
nlp.close()

def dataPreprocessForGCN(text):

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
    idx = [[], []]
    for i in range(len(idx1)):
        idx[0].append(idx2[i])
        idx[1].append(idx1[i])

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

    # 添加文本顺序特征
    for i in range(len(sentence) - 1):
        adjacency_matrix[i, i+1] = 1
        # adjacency_matrix[i+1, i] = 1
    # 添加分词长度前缀和数组
    words_len = []
    k = 0
    for tmp in words:
        if k == 0:
            words_len.append(0)
        words_len[k] = words_len[k-1] + len(tmp)

    # 从边数组中获取边的数量
    num_edges = edge_index.shape[1]         ####    [2, num_edges]
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
    print(node_degrees)

    # 计算度矩阵的逆平方根
    degree_inv_sqrt = np.power(node_degrees, -0.5)
    degree_inv_sqrt_matrix = np.diag(degree_inv_sqrt)

    # 对邻接矩阵进行归一化
    adj_normalized = degree_inv_sqrt_matrix @ adjacency_matrix @ degree_inv_sqrt_matrix

    # 现在 adj_normalized 是归一化后的邻接矩阵
    print(adj_normalized)


    print("data_process________ADJ_____Normalized")

        # 打印邻接矩阵（可选）

    print(adj_normalized.shape)

    print(adj_normalized)

    # print(adj)

    #edge_index -> adj, data -> input_feature
    return adj_normalized, words
    # return adj,words
    # return idx, words
