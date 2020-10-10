# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable
from utils import data_helpers as dh


class FVModel(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, config):
        super(FVModel, self).__init__()

        # Model configuration
        self.config = config

        # cpu or gpu
        self.device = torch.device("cuda" if (self.cuda and torch.cuda.is_available()) else "cpu")
        print(self.device)

        # Layer definitions
        # Item embedding layer, item's index
        self.encode = torch.nn.Embedding(num_embeddings=config.num_product,
                                        embedding_dim=config.embedding_dim,
                                        padding_idx=0)
        self.weight_change_encode = torch.nn.Embedding(num_embeddings=2,
                                        embedding_dim=config.embedding_dim_weight,
                                        padding_idx=0)  # weight_change just has 3 values
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            # getattr() 获取对象属性
            self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim + config.embedding_dim_weight,
                                                        hidden_size=config.embedding_dim + config.embedding_dim_weight,
                                                        num_layers=config.rnn_layer_num,
                                                        batch_first=True,
                                                        dropout=config.dropout,
                                                        bidirectional=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.embedding_dim + config.embedding_dim_weight,
                                    hidden_size=config.embedding_dim + config.embedding_dim_weight,
                                    num_layers=config.rnn_layer_num,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=False)

        self.fc = torch.nn.Linear(config.embedding_dim + config.embedding_dim_weight, config.embedding_dim)

    def forward(self, x, lengths, hidden):
        length = lengths[0]
        # Basket Encoding
        # users' basket sequence
        # x: uids, baskets, lens
        ub_seqs = torch.Tensor(self.config.batch_size, length, self.config.embedding_dim + self.config.embedding_dim_weight).to(self.device)
        # uw_seqs = torch.LongTensor(self.config.batch_size, length, 1).to(self.device)  # [256, 30, 1]
        for (i, user) in enumerate(x):  # shape of x: [batch_size, seq_len, indices of product]
            embed_baskets = torch.Tensor(length, self.config.embedding_dim + self.config.embedding_dim_weight).to(self.device)
            # uw = torch.LongTensor(length, 1).to(self.device)  # [30, 1]
            for (j, basket) in enumerate(user):  # shape of user: [seq_len, indices of product]
                if basket[-1] == 1:
                    weight = [1]  # task1 true label  [1]
                else:
                    weight = [0]
                weight = torch.LongTensor(weight).resize_(1, len(weight)).to(self.device)  # shape: [1,1]   # not LongTensor??
                weight = self.weight_change_encode(torch.autograd.Variable(weight))
                weight = weight.reshape(self.config.embedding_dim_weight)
                basket = basket[:-1]
                basket = torch.LongTensor(basket).resize_(1, len(basket)).to(self.device)
                basket = self.encode(torch.autograd.Variable(basket))  # shape: [1, len(basket), embedding_dim]
                basket = self.pool(basket, dim=1)
                basket = basket.reshape(self.config.embedding_dim)
                basket = torch.cat([basket, weight])  # concat Variety change
                #embed_baskets[j] = basket  # shape:  [seq_len, 1, embedding_dim]
                embed_baskets[j] = basket  # shape:  [seq_len, 1, embedding_dim]

                
            # Concat current user's all baskets and append it to users' basket sequence
            ub_seqs[i] = embed_baskets  # shape: [batch_size, seq_len, embedding_dim]

        # Packed sequence as required by pytorch
        packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)

        # RNN
        output, h_u = self.rnn(packed_ub_seqs, hidden)

        # shape: [batch_size, true_len(before padding), embedding_dim]
        out2, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        dynamic_user = self.fc(out2)
        # print("dynamic_user.shape:   ", dynamic_user.shape)

        return dynamic_user, h_u

    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        self.encode.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim + self.config.embedding_dim_weight).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim + self.config.embedding_dim_weight).zero_()))
        else:
            return Variable(torch.zeros(self.config.rnn_layer_num, batch_size, self.config.embedding_dim + self.config.embedding_dim_weight).to(self.device))
