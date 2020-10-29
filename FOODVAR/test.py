# -*- coding:utf-8 -*-

import time
import random
import math
import pickle
import torch
import numpy as np
from config import Config
from utils import data_helpers as dh
from tqdm import tqdm
from scipy.stats import ttest_ind

logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime()))

# MODEL = input("☛ Please input the model file you want to test: ")

# while not (MODEL.isdigit() and len(MODEL) == 10):
#     MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
# logger.info("✔︎ The format of your input is legal, now loading to next step...")

# MODEL_DIR = dh.load_model_file(MODEL)


def test(saved_file):
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data(Config().TESTSET_DIR)

    logger.info("✔︎ Load negative sample...")
    # with open(Config().NEG_SAMPLES, 'rb') as handle:
    #     neg_samples = pickle.load(handle)
    neg_samples = {}

    item_list = [i for i in range(336)]

    # Load model
    MODEL_DIR = dh.load_model_file(saved_file)

    dr_model = torch.load(MODEL_DIR)

    dr_model.eval()

    item_embedding = dr_model.encode.weight
    hidden = dr_model.init_hidden(Config().batch_size)

    hitratio_numer = 0
    hitratio_denom = 0
    hitratio_numer = 0
    hitratio_denom = 0
    hitratio_numer_10 = 0
    hitratio_numer_5 = 0
    ndcg = 0.0
    ndcg_denom = 0
    hitratio_list_5 = []
    hitratio_list_10 = []
    ndcg_list = []


    for i, x in enumerate(tqdm(dh.batch_iter(train_data, Config().batch_size, Config().seq_len_test, shuffle=False))):
        uids, baskets, lens = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du in zip(uids, lens, dynamic_user):
            scores = []
            du_latest = du[l - 1].unsqueeze(0)

            # Deal with positives samples            
            positives = test_data[test_data['userID'] == uid].baskets.values[0][:-1]  # list dim 1
            p_length = len(positives)
            positives = torch.LongTensor(positives)
            print("positives:   ", positives)
            
            # calculating <u,p> score for all test items <u,p> pair
            scores_pos = list(torch.mm(du_latest, item_embedding[positives].t()).data.cpu().numpy()[0])
            for s in scores_pos:
                scores.append(s)
            print("score_pos:   ", score_pos)

            # Deal with negative samples
            neg_item_list = list(set(item_list).difference(set(positives)))
            negtives = random.sample(neg_item_list, Config().neg_num)
            negtives = torch.LongTensor(negtives)
            scores_neg = list(torch.mm(du_latest, item_embedding[negtives].t()).data.cpu().numpy()[0])
            for s in scores_neg:
                scores.append(s)
            
            print("scores:   ", scores)

            # Calculate hit-ratio
            index_k = []
            for k in range(Config().top_k):
                index = scores.index(max(scores))
                index_k.append(index)
                scores[index] = -9999
            print("index_k:   ", index_k)
            hr_5_numer = len((set(np.arange(0, p_length)) & set(index_k[0:5])))
            hr_10_numer = len((set(np.arange(0, p_length)) & set(index_k)))
            hitratio_numer_10 += hr_10_numer  # np.arange()产生等差数列
            hitratio_numer_5 += hr_5_numer
            hitratio_denom += p_length
            hitratio_list_5.append(hr_5_numer/p_length)
            hitratio_list_10.append(hr_10_numer/p_length)
            # print("hitratio_list_5:   ", hitratio_list_5)
            # print("hitratio_list_10:   ", hitratio_list_10)
            # hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
            # hitratio_denom += p_length

            # Calculate NDCG
            u_dcg = 0
            u_idcg = 0
            for k in range(Config().top_k):
                if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                    u_dcg += 1 / math.log(k + 1 + 1, 2)
                u_idcg += 1 / math.log(k + 1 + 1, 2)
            ndcg += u_dcg / u_idcg
            ndcg_denom += 1
            ndcg_list.append(u_dcg / u_idcg)
            # print("ndcg_list:   ", ndcg_list)

    hit_ratio_5 = hitratio_numer_5 / hitratio_denom
    hit_ratio_10 = hitratio_numer_10 / hitratio_denom
    ndcg = ndcg / ndcg_denom
    print('Hit ratio@5: {1} | Hit ratio@10: {1}'.format(hit_ratio_5, hit_ratio_10))
    print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    return hitratio_list_5, hitratio_list_10, ndcg_list


if __name__ == '__main__':
    # increase: baseline/1598233978
    # decrease: baseline/1598234060
    # maintenance: baseline/1598238424
    # increase: foodvar/1602738103
    # decrease: foodvar/1602738184
    # maintenance: foodvar/1602770086
    in_bas_hr_5, in_bas_hr_10, in_bas_ndcg = test("baseline/1598233978")
    # de_bas_hr_5, de_bas_hr_10, de_bas_ndcg = test("baseline/1598234060")
    # ma_bas_hr_5, ma_bas_hr_10, ma_bas_ndcg = test("baseline/1598238424")

    in_fv_hr_5, in_fv_hr_10, in_fv_ndcg = test("foodvar/1602738103")
    # de_fv_hr_5, de_fv_hr_10, de_fv_ndcg = test("foodvar/1602738184")
    # ma_fv_hr_5, ma_fv_hr_10, ma_fv_ndcg = test("foodvar/1602770086")

    # a = [ma_fv_hr_5, ma_fv_hr_10, ma_fv_ndcg, de_fv_hr_5, de_fv_hr_10, de_fv_ndcg, in_fv_hr_5, in_fv_hr_10, in_fv_ndcg]
    # b = [ma_bas_hr_5, ma_bas_hr_10, ma_bas_ndcg, de_bas_hr_5, de_bas_hr_10, de_bas_ndcg, in_bas_hr_5, in_bas_hr_10, in_bas_ndcg]

    # for i in range(len(a)):
    #     _, p = ttest_ind(a[i], b[i], equal_var = False)
    #     # _, p = ttest_1samp(a[i], b[i])
    #     print(p)

    # hitratio_list_5, hitratio_list_10, ndcg_list = test(saved_file)
    # print("hitratio_list_5:   ", hitratio_list_5)
    # print("hitratio_list_10:   ", hitratio_list_10)
    # print("ndcg_list:   ", ndcg_list)
