import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import dgl
import scipy.sparse as sp
import pickle

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from load_data import *
from util import *
from ssg_modules import gru_module
from ssg_args import get_parser
from subdata import SubDataset

parser = get_parser()
opts = parser.parse_args()

ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/'
dataset_name = 'CD_5'
dataset_path = ROOT+'dataset/{}/{}.json'.format(dataset_name, dataset_name)
save_path = ROOT + 'dataset/{}/'.format(dataset_name)
beta_lst = [0.0, 1.0]


class Attn_cal(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size, beta,
                 symm=True, mix_cpu_gpu=True, use_user_item_doc=False):
        self._device = device
        self._review_fea_size = review_fea_size
        self._symm = symm
        self.beta = beta

        review_feat_path = \
             f'{ROOT}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'

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

            return user_id, item_id, rating, time

        self.train_datas = process_sent_data(sent_train_data)
        self.possible_rating_values = np.unique(self.train_datas[2])


        self.user_feature = None
        self.movie_feature = None


        train_rating_pairs, train_rating_values, train_time_values = self._generate_pair_value('train')

        self._generate_score(train_rating_pairs,
                                                        train_rating_values,
                                                        add_support=True)


    

    def _generate_pair_value(self, sub_dataset):
        user_id = self.train_datas[0]
        item_id = self.train_datas[1]
        rating = self.train_datas[2]
        time = self.train_datas[3]

        rating_pairs = (np.array(user_id, dtype=np.int64),
                        np.array(item_id, dtype=np.int64))
        rating_values = np.array(rating, dtype=np.float32)
        time_values = np.array(time, dtype=np.float32)
        return rating_pairs, rating_values, time_values
    
    
        


#################
    def _generate_score(self, rating_pairs, rating_values,
                            add_support=False):
        
        seq_model = gru_module(input_dim=opts.input_dim, gru_dim=opts.gru_dim, time_dim=opts.time_dim, beta=self.beta) # 64, 32, 32, 1.0
        seq_model.cuda()

        dataset = SubDataset(
                            dataset_name,
                            dataset_path,
                            'cpu',
                            review_fea_size=opts.emb_dim, # 64
                            symm=True,
                            )
        # user_ids = dataset.train_datas[0]
        u_text_s = dataset.u_text_s
        i_text_s = dataset.i_text_s
        u_dt_dic = dataset.u_dt_dic
        i_dt_dic = dataset.i_dt_dic

        rating_row, rating_col = rating_pairs

        u_attn = []
        i_attn = []
        print('u_gru,,,')
        for i in range(len(rating_row)):
            u_id = rating_row[i]
            i_id = rating_col[i]
            i_id_ind = u_text_s[u_id]['item_id'].index(i_id)
            reviews_u = u_text_s[u_id]['review']
            reviews_u = torch.tensor(reviews_u).unsqueeze(0).cuda() 
            u_s_renum = u_dt_dic[u_id]['u_s_renum'] # int
            u_s_renum = torch.tensor([u_s_renum]).cuda()
            u_pos_ind = u_dt_dic[u_id]['u_pos_ind']
            u_pos_ind = torch.tensor(u_pos_ind).unsqueeze(0).cuda()
            u_rel_dt = u_dt_dic[u_id]['u_rel_dt']
            u_rel_dt = torch.tensor(u_rel_dt).unsqueeze(0).cuda()
            u_abs_dt = u_dt_dic[u_id]['u_abs_dt']
            u_abs_dt = torch.tensor(u_abs_dt).unsqueeze(0).cuda()
            u_hn = seq_model(reviews_u, u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt)
            u_hn = u_hn.view(-1)
            if len(u_hn) > 1:
                u_hn = u_hn[i_id_ind]
            else: 
                u_hn = u_hn[0]
            u_attn.append(u_hn.cpu())
            # u_attn.append(u_hn.item())
        u_attn = torch.tensor(u_attn)
        
        print('i_gru,,,')
        for i in tqdm(range(len(rating_col))):
            i_id = rating_col[i]
            u_id = rating_row[i]
            u_id_ind = i_text_s[i_id]['user_id'].index(u_id)

            reviews_i = i_text_s[i_id]['review']
            reviews_i = torch.tensor(reviews_i).unsqueeze(0).cuda() 
            i_s_renum = i_dt_dic[i_id]['i_s_renum'] # int
            i_s_renum = torch.tensor([i_s_renum]).cuda()
            i_pos_ind = i_dt_dic[i_id]['i_pos_ind']
            i_pos_ind = torch.tensor(i_pos_ind).unsqueeze(0).cuda()
            i_rel_dt = i_dt_dic[i_id]['i_rel_dt']
            i_rel_dt = torch.tensor(i_rel_dt).unsqueeze(0).cuda()
            i_abs_dt = i_dt_dic[i_id]['i_abs_dt']
            i_abs_dt = torch.tensor(i_abs_dt).unsqueeze(0).cuda()
            i_hn = seq_model(reviews_i, i_s_renum, i_pos_ind, i_rel_dt, i_abs_dt)
            i_hn = i_hn.view(-1)
            if len(i_hn) > 1:
                i_hn = i_hn[u_id_ind]
            else: 
                i_hn = i_hn[0]
            i_attn.append(i_hn.cpu())
            # i_attn.append(i_hn.item())
        i_attn = torch.tensor(i_attn)

        print('attn_len', len(i_attn))

        with open(save_path + f'u_attn{self.beta}.pickle', 'wb') as f:
            pickle.dump(u_attn, f, pickle.HIGHEST_PROTOCOL)

        with open(save_path + f'i_attn{self.beta}.pickle', 'wb') as f:
            pickle.dump(i_attn, f, pickle.HIGHEST_PROTOCOL)
       






if __name__ == '__main__':
    for beta in beta_lst:
        print('beta', beta)

        dataset = Attn_cal(dataset_name,
                            dataset_path,
                            2,
                            review_fea_size=64,
                            beta=beta,
                            symm=True)



