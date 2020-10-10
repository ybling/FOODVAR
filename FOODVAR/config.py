# -*- coding:utf-8 -*-

import torch

class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = './data/train_sample.json'
        self.VALIDATIONSET_DIR = './data/validation_sample.json'
        self.TESTSET_DIR = './data/test_sample.json'
        # self.NEG_SAMPLES = './data/neg_sample.pickle'
        self.MODEL_DIR = './runs/'
        self.cuda = True
        self.clip = 1
        self.epochs = 50
        self.batch_size = 256
        self.seq_len_train = 30 # max sequence len, use padding here     padding len   # 数据长度一致 不需要padding
        self.seq_len_valid = 7 # max sequence len, use padding here     padding len   # 数据长度一致 不需要padding
        self.seq_len_test = 1 # max sequence len, use padding here     padding len   # 数据长度一致 不需要padding
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 8  # num of batches between two logging
        self.basket_pool_type = 'avg'  # ['avg', 'max']
        self.rnn_type = 'LSTM'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        self.dropout = 0.5
        self.num_product = 336 # 商品数目，用于定义 Embedding Layer  
        self.embedding_dim = 32  # 商品表征维数， 用于定义 Embedding Layer
        self.embedding_dim_weight = 3  # 商品表征维数，用于定义 Embedding Layer

        self.neg_num = 50  # the number of negative sampling
        self.top_k = 10  # Top K 
        self.loss_weight = 0.2

        # config for rnn_weight model
        self.rnn_layer_num_1 = 2 
        self.output_size_1 = 3


