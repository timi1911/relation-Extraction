import torch
from stanfordcorenlp import StanfordCoreNLP


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
    #edge_index -> adj, data -> input_feature
    return edge_index, words

text = "蔡志坚在南京艺术学院求学时受过系统、正规的艺术教育和专业训练，深得刘海粟、罗叔子、陈之佛、谢海燕、陈大羽等著名中国画大师的指授，基本功扎实，加上他坚持从生活中汲取创作源泉，用心捕捉生活中最美最感人的瞬间形象，因而他的作品，不论是山水、花鸟、飞禽、走兽，无不充满了生命的灵气，寄托着画家的情怀，颇得自然之真趣。"
print(dataPreprocessForGCN(text))