import numpy as np
from stanfordnlp.server import CoreNLPClient


def dependency_parse_to_adjacency_matrix(sentence):
    # 创建CoreNLP客户端
    with CoreNLPClient(annotators='depparse', timeout=30000, memory='4G') as client:
        # 进行依存句法分析
        ann = client.annotate(sentence, output_format='json')

        # 获取依存关系结果
        dependencies = ann['sentences'][0]['basicDependencies']

        # 创建256*256的邻接矩阵
        adjacency_matrix = np.zeros((256, 256))

        # 将依存关系结果转化为邻接矩阵
        for dep in dependencies:
            gov_index = dep['governor']
            dep_index = dep['dependent']
            adjacency_matrix[gov_index - 1][dep_index - 1] = 1

        return adjacency_matrix


# 测试示例
sentence = '我喜欢吃水果。'
adjacency_matrix = dependency_parse_to_adjacency_matrix(sentence)
print(adjacency_matrix)