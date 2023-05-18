import torch
import torch.nn as nn
import torch.utils.data as data

import os, sys, copy
import numpy as np
import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from load_data import *

ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/'

class SubDataset(object):
    def __init__(self, dataset_name, dataset_path, device, review_fea_size,
                 symm=True, mix_cpu_gpu=True, use_user_item_doc=False):
        self._device = device
        self._review_fea_size = review_fea_size
        self._symm = symm

        review_feat_path = \
             f'{ROOT}BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'

        try:
            self.train_review_feat = torch.load(review_feat_path)
        except FileNotFoundError:
            self.train_review_feat = None
            print(f'Load pretrained review feature fail! feature size:{review_fea_size}')

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = \
                load_sentiment_data(dataset_path)

        self.word2id = None
        self.embedding = None
        self.user_doc = None
        self.movie_doc = None

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            item_id = info['item_id'].to_list()
            rating = info['rating'].to_list()
            time = info['time'].to_list()

            return user_id, item_id, rating, time#, review_text

        self.train_datas = process_sent_data(sent_train_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self._num_user = dataset_info['user_size']
        self._num_movie = dataset_info['item_size']

        self.user_feature = None
        self.movie_feature = None

        self.user_feature_shape = (self.num_user, self.num_user)
        self.movie_feature_shape = (self.num_movie, self.num_movie)

        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        self.train_rating_pairs, train_rating_values, train_time_values = self._generate_pair_value('train')

        def get_review_feature(rating_pairs):
            record_size = rating_pairs[0].shape[0]
            review_feat_list = [self.train_review_feat[(rating_pairs[0][x], rating_pairs[1][x])] for x in range(record_size)]
            return review_feat_list  # user_movie_r, 
        
        self.train_review_feat = get_review_feature(self.train_rating_pairs) # 임베딩 리뷰
        
        self.u_text = dict()
        self.i_text = dict()

        for i in range(len(self.train_datas[0])): #  user_id, item_id, rating, time#
            d = self.train_datas
            if d[0][i] not in self.u_text:
                self.u_text[d[0][i]] = {'item_id': [d[1][i]], 'time': [d[3][i]], 'review': [self.train_review_feat[i].tolist()]}
            else:
                self.u_text[d[0][i]]['item_id'].append(d[1][i])
                self.u_text[d[0][i]]['time'].append(d[3][i])
                self.u_text[d[0][i]]['review'].append(self.train_review_feat[i].tolist())

            if d[1][i] not in self.i_text:
                self.i_text[d[1][i]] = {'user_id': [d[0][i]], 'time': [d[3][i]], 'review': [self.train_review_feat[i].tolist()]}
            else:
                self.i_text[d[1][i]]['user_id'].append(d[0][i])
                self.i_text[d[1][i]]['time'].append(d[3][i])
                self.i_text[d[1][i]]['review'].append(self.train_review_feat[i].tolist())
        self.u_text_s = copy.deepcopy(self.u_text)
        self.i_text_s = copy.deepcopy(self.i_text)
        # 시간 순 정렬
        for i in self.u_text_s: # user_id
            sorted_ind = np.argsort(self.u_text_s[i]['time'])
            self.u_text_s[i]['time'] = np.sort(self.u_text_s[i]['time'])
            self.u_text_s[i]['review'] = [self.u_text_s[i]['review'][j] for j in sorted_ind]
            self.u_text_s[i]['item_id'] = [self.u_text_s[i]['item_id'][j] for j in sorted_ind]
            self.u_text_s[i]['sorted_ind'] = sorted_ind
        for i in self.i_text_s:
            sorted_ind = np.argsort(self.i_text_s[i]['time'])
            self.i_text_s[i]['time'] = np.sort(self.i_text_s[i]['time'])
            self.i_text_s[i]['review'] = [self.i_text_s[i]['review'][j] for j in sorted_ind]
            self.i_text_s[i]['user_id'] = [self.i_text_s[i]['user_id'][j] for j in sorted_ind]
            self.i_text_s[i]['sorted_ind'] = sorted_ind

        self.u_dt_dic = dict()
        self.i_dt_dic = dict()
        for i in self.u_text_s:
            u_times = self.u_text_s[i]['time']
            for j in range(len(self.u_text_s[i]['time'])):
                time = self.u_text_s[i]['time'][j]
                u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt = self.time_handler(
                u_times, time)
                self.u_dt_dic[i] = {'u_s_renum': u_s_renum, 'u_pos_ind': u_pos_ind, 'u_rel_dt': u_rel_dt, 'u_abs_dt': u_abs_dt}
            # if i== 1 : 
            #     print(self.u_dt_dic[i], u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt)
            #     break
        
        for i in self.i_text_s:
            i_times = self.i_text_s[i]['time']
            for j in range(len(self.i_text_s[i]['time'])):
                time = self.i_text_s[i]['time'][j]
                i_s_renum, i_pos_ind, i_rel_dt, i_abs_dt = self.time_handler(
                i_times, time)
                self.i_dt_dic[i] = {'i_s_renum': i_s_renum, 'i_pos_ind': i_pos_ind, 'i_rel_dt': i_rel_dt, 'i_abs_dt': i_abs_dt}
            

        #     u_times = self.u_text_s[i]['time']
        #     u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt = self.time_handler(
        #     u_times, time)
        # print(self.train_rating_pairs[0])

        # print(self.u_text[2]['review'])
        # print(len(self.u_text[2]['review']))

    # 계산 추가
    def time_handler(self, times, cur_time): # cur_time : 보고자하는
        percentile =10
        max_rel = 100
        times = list(times.reshape(-1))
        renum = 0
        pos_ind = []
        rel_dt = []
        abs_dt = []
        renum_lst = []
        for ind, t in enumerate(times):
            # if t < cur_time: # renum이 0이되는 문제가 발생하므로,,,수정
            if t < cur_time or ind == 0:
                dt = cur_time - t
                rel_dt.append(dt) 
                abs_dt.append(dt)# 절대적 거리 계산 ti
                pos_ind.append(renum) # 절대적 거리 계산한 인덱스
                renum += 1 # 인덱스 개수
            else:
                rel_dt.append(0)
                abs_dt.append(0)
                pos_ind.append(0)

        for i in range(len(pos_ind)): # pos_ind 내림차순 유사하게 번경
            if i < renum:
                pos_ind[i] = renum - pos_ind[i]

        dt = []
        for i in range(len(rel_dt)):
            if i == 0:
                continue
            dt.append(rel_dt[i - 1] - rel_dt[i]) # interval 계산해서 dt에 저장
            if rel_dt[i] == 0:
                break
        dt = np.array(dt)

        pos_ind = np.array(pos_ind)
        rel_dt = np.array(rel_dt)
        abs_dt = np.array(abs_dt)
        if any(rel_dt):
            dt_nonzero = np.delete(dt, np.where(dt == 0))
            m = max(np.percentile(dt_nonzero, percentile), 1.0)
            for i in range(len(rel_dt)):
                rel_dt[i] = min((rel_dt[i] // m), max_rel) ###  interval 조정
        return renum, pos_ind, rel_dt, abs_dt

    def _generate_pair_value(self, sub_dataset):
        user_id = self.train_datas[0]
        item_id = self.train_datas[1]
        rating = self.train_datas[2]
        time = self.train_datas[3]
        # review = self.train_datas[4]

        rating_pairs = (np.array(user_id, dtype=np.int64),
                        np.array(item_id, dtype=np.int64))
        rating_values = np.array(rating, dtype=np.float32)
        time_values = np.array(time, dtype=np.float32)
        # review_values = np.array(review, dtype=str)

        return rating_pairs, rating_values, time_values # , review_values
    

    
    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

