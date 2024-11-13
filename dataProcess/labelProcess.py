import json
from random import choice
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
import torch
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


def make_entity_label(source, label_list ,target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            for j in range(target_len):
                label_list[i + j] = 1
    return label_list

def find_word_label(source, label_list ,target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            for j in range(len(target)):
                if label_list[i + j] != 1:
                    return 0
    return 1


