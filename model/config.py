import json


class Config(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_name = args.bert_name
        self.bert_dim = args.bert_dim

        self.train_path = 'data/' + self.dataset + '/train.json'
        self.test_path = 'data/' + self.dataset + '/test.json'
        self.dev_path = 'data/' + self.dataset + '/dev.json'
        self.rel_path = 'data/' + self.dataset + '/rel.json'
        #此处将'r'修改成了'rb'
        print(self.rel_path)
        self.num_relations = len(json.load(open(self.rel_path, 'rb')))

        self.save_weights_dir = 'saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset + '/'

        self.period = 200
        self.test_epoch = 3
        self.weights_save_name = 'model.pt'
        self.log_save_name = 'model.out'
        self.result_save_name = 'result.json'


{"text": "燕先知，女，汉族，大学", "spo_list": [{"predicate": "民族", "object_type": "文本", "subject_type": "人物", "object": "汉族", "subject": "燕先知"}]}
{"text": "流纹岩分布较少，多呈透镜状分布。", "spo_list": [{"predicate": "分布形态", "object_type": "岩石", "subject_type": "形状", "object": "流纹岩", "subject": "透镜状"}]}

