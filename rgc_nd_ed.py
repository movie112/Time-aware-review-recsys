# -*- coding: utf-8 -*-

import os
import torch 
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import argparse
from abc import ABC
import csv

from data import MovieLens
from util import *

import random
import string
# import math
# from collections import defaultdict
ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/'
def config():
    parser = argparse.ArgumentParser(description='RGC-ND-ED')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    # parser.add_argument('--model_save_path', type=str,
    #                     help='The model saving path')
    parser.add_argument('-dn', '--dataset_name', type=str,
                        help='dataset name')
    parser.add_argument('-dp', '--dataset_path', type=str,
                        help='raw dataset file path')

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=20)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)##
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--train_classification', type=bool, default=False)
    parser.add_argument('--review_feat_size', type=int, default=64)

    parser.add_argument('--rating_alpha', type=float, default=1.)
    parser.add_argument('--ed_alpha', type=float, default=.1)
    parser.add_argument('--nd_alpha', type=float, default=1.)
    parser.add_argument('--distributed', type=bool, default=False)

    parser.add_argument('--model_short_name', type=str, default='RGC-ND-ED')
    parser.add_argument('--logger', type=str, default='logger')
    parser.add_argument('--logger_file', type=str, default='')
    parser.add_argument('--root_path', type=str, default=ROOT)
    parser.add_argument('--beta', type=str, default=1.0)

    args = parser.parse_args()

    if not os.path.isdir(f'{args.root_path}log'):
        os.makedirs(f'{args.root_path}log')

    if not os.path.isdir('log'):
        os.makedirs('log')

    args.gcn_agg_units = args.review_feat_size
    args.gcn_out_units = args.review_feat_size

    return args


class GCMCGraphConv(nn.Module, ABC):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        self.prob_score = nn.Linear(self._out_feats, 1, bias=False)
        self.review_score = nn.Linear(self._out_feats, 1, bias=False)
        # self.prob_score = nn.Linear(1, 1, bias=True)
        # self.review_score = nn.Linear(1, 1, bias=True)
        self.review_w = nn.Linear(out_feats, out_feats, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.prob_score.weight)
        init.xavier_uniform_(self.review_score.weight)
        init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph, feat, weight=None): #####

        with graph.local_scope():
            graph.edata['attn'] = graph.edata['attn'].view(-1, 1)

            feat = self.weight

            graph.srcdata['h'] = feat
            review_feat = graph.edata['review_feat']

            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
            graph.edata['ra'] = torch.sigmoid(self.review_score(review_feat))
            graph.edata['rf'] = self.review_w(review_feat)
            graph.update_all(lambda edges: {'m': (edges.src['h'] * (edges.data['pa'])
                                                  + edges.data['rf'] * ((edges.data['ra'])*graph.edata['attn']))
                                                 * self.dropout(edges.src['cj'])},
                             fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h'] * graph.dstdata['ci']

        return rst 


class GCMCLayer(nn.Module, ABC):

    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 out_units,
                 dropout_rate=0.0,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.ufc = nn.Linear(out_units, out_units)
        self.ifc = nn.Linear(out_units, out_units)
        self.dropout = nn.Dropout(dropout_rate)
        sub_conv = {}
        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            sub_conv[rating] = GCMCGraphConv(user_in_units,
                                             out_units,
                                             device=device,
                                             dropout_rate=dropout_rate)
            sub_conv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                 out_units,
                                                 device=device,
                                                 dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate='sum')
        self.agg_act = nn.LeakyReLU(0.1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user': ufeat, 'movie': ifeat}
        out_feats = self.conv(graph, in_feats)
        ufeat = out_feats['user']
        ifeat = out_feats['movie']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return ufeat, ifeat


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        init.xavier_uniform_(self.w.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y ).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss

    def measure_sim(self, x, y):
        if len(y.shape) > len(x.shape):
            _l = y.shape[1]
            _x = x @ self.w
            _x = _x.unsqueeze(1)
            return (_x * y).sum(-1)

        else:
            return (x @ self.w * y_neg).sum(-1)


class MLPPredictorMI(nn.Module, ABC):
    """
    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self,
                 in_units,
                 num_classes,
                 dropout_rate=0.0,
                 neg_sample_size=1):
        super(MLPPredictorMI, self).__init__()
        self.neg_sample_size = neg_sample_size
        self.dropout = nn.Dropout(dropout_rate)

        self.contrast_loss = ContrastLoss(in_units)

        self.linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.predictor = nn.Linear(in_units, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def neg_sampling(graph):
        review_feat = graph.edata['review_feat']
        neg_review_feat = review_feat[torch.randperm(review_feat.shape[0]), :]
        return neg_review_feat

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_fea = self.linear(torch.cat([h_u, h_v], dim=1))
        score = self.predictor(h_fea).squeeze()

        if 'neg_review_feat' in edges.data:
            review_feat = edges.data['review_feat']
            # neg_review_feat = review_feat[torch.randperm(review_feat.shape[0]), :]
            neg_review_feat = edges.data['neg_review_feat']
            mi_score = self.contrast_loss(h_fea, review_feat, neg_review_feat)
            return {'score': score, 'mi_score': mi_score}
        else:
            return {'score': score}

    def forward(self, graph, ufeat, ifeat, cal_edge_mi=True):
        graph.nodes['user'].data['h'] = ufeat
        graph.nodes['movie'].data['h'] = ifeat

        if ('review_feat' in graph.edata) & cal_edge_mi:
            graph.edata['neg_review_feat'] = self.neg_sampling(graph)
        else:
            del graph.edata['neg_review_feat']

        with graph.local_scope():
            graph.apply_edges(self.apply_edges)
            if 'mi_score' in graph.edata:
                return graph.edata['score'], graph.edata['mi_score']
            else:
                return graph.edata['score']


class Net(nn.Module, ABC):
    def __init__(self, params):
        super(Net, self).__init__()
        self._params = params
        self.encoder = GCMCLayer(params.rating_vals,
                                 params.src_in_units,
                                 params.dst_in_units,
                                 params.gcn_out_units,
                                 dropout_rate=params.gcn_dropout,
                                 device=params.device)

        if params.train_classification:
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=len(params.rating_vals))
        else: 
            self.decoder = MLPPredictorMI(in_units=params.gcn_out_units,
                                        num_classes=1)
        # self.contrast_loss = ContrastLoss(params.gcn_out_units, x_inner=True)
        self.contrast_loss = ContrastLoss(params.gcn_out_units)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat, cal_edge_mi=True):

        user_out, movie_out = self.encoder(enc_graph, ufeat, ifeat)
        if self._params.distributed:
            user_out = user_out.to(self._params.device)
            movie_out = movie_out.to(self._params.device)

        if cal_edge_mi:
            pred_ratings, mi_score = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi)
            return pred_ratings, mi_score, user_out, movie_out
        else:
            pred_ratings = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi)
            return pred_ratings, user_out, movie_out


def evaluate(params, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    if params.distributed:
        nd_possible_rating_values = torch.FloatTensor(possible_rating_values).to(params.device)
        rating_values = rating_values.to(params.device)
    else:
        nd_possible_rating_values = torch.FloatTensor(possible_rating_values).to(params.device)

    # Evaluate RMSE
    net.eval()
    with torch.no_grad():
        pred_ratings, _, _, _ = net(enc_graph, dec_graph,
                                    dataset.user_feature, dataset.movie_feature)
        if params.train_classification:
            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) *
                                 nd_possible_rating_values.view(1, -1)).sum(dim=1)
            rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
        else:
            rmse = ((pred_ratings - rating_values) ** 2.).mean().item()
        # rmse = np.sqrt(rmse)
    return rmse


def train(params):

    print(params)
    logger.info(params)
    dataset = MovieLens(params.dataset_name,
                        params.dataset_path,
                        params.device,
                        params.review_feat_size,
                        params.beta,
                        symm=True)
    print("Loading data finished ...\n")

    params.src_in_units = dataset.user_feature_shape[1]
    params.dst_in_units = dataset.movie_feature_shape[1]
    params.rating_vals = dataset.possible_rating_values

    net = Net(params)

    if params.distributed:
        net.encoder.cpu()
        net.decoder.to(params.device)
        net.contrast_loss.to(params.device)
        nd_possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(params.device)
    else:
        net = net.to(params.device)
        nd_possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(params.device)

    rating_loss_net = nn.CrossEntropyLoss() if params.train_classification else nn.MSELoss()
    learning_rate = params.train_lr
    # optimizer = get_optimizer(params.train_optimizer)(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    # prepare training data
    if params.train_classification:
        train_gt_labels = dataset.train_labels
        train_gt_ratings = dataset.train_truths
    else:
        train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

    if params.distributed:
        train_gt_labels = train_gt_labels.to(params.device)
        train_gt_ratings = train_gt_ratings.to(params.device)

    # declare the loss information
    # best_valid_rmse = np.inf
    # best_test_rmse = np.inf
    best_valid_mse = np.inf
    best_test_mse = np.inf
    no_better_valid = 0
    best_iter = -1

    if params.distributed:
        dataset.train_enc_graph = dataset.train_enc_graph.int().cpu()
        dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
        dataset.valid_enc_graph = dataset.train_enc_graph
        dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
        dataset.test_enc_graph = dataset.test_enc_graph.int().cpu()
        dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    else:
        dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)
        dataset.train_dec_graph = dataset.train_dec_graph.int().to(params.device)
        dataset.valid_enc_graph = dataset.train_enc_graph
        dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(params.device)
        dataset.test_enc_graph = dataset.test_enc_graph.int().to(params.device)
        dataset.test_dec_graph = dataset.test_dec_graph.int().to(params.device)

    print("Start training ...")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        optimizer.zero_grad()

        pred_ratings1, ed_mi1, user1, item1 = net(dataset.train_enc_graph,
                                                  dataset.train_dec_graph,
                                                  dataset.user_feature,
                                                  dataset.movie_feature,)
        pred_ratings2, ed_mi2, user2, item2 = net(dataset.train_enc_graph,
                                                  dataset.train_dec_graph,
                                                  dataset.user_feature,
                                                  dataset.movie_feature)
        loss1 = rating_loss_net(pred_ratings1, train_gt_labels).mean()
        loss2 = rating_loss_net(pred_ratings2, train_gt_labels).mean()
        user_mi_loss = net.contrast_loss(user1, user2).mean()
        item_mi_loss = net.contrast_loss(item1, item2).mean()
        r_loss = (loss1 + loss2) / 2

        nd_loss = (user_mi_loss + item_mi_loss) / 2
        # nd_loss = item_mi_loss
        ed_loss = (ed_mi1.mean() + ed_mi2.mean()) / 2

        total_loss = r_loss + params.nd_alpha * nd_loss + params.ed_alpha * ed_loss
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
        optimizer.step()

        if params.train_classification:
            real_pred_ratings = (torch.softmax(pred_ratings1, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        else: 
            real_pred_ratings = pred_ratings1

        # train_rmse = ((real_pred_ratings - train_gt_ratings) ** 2).mean().sqrt()
        train_mse = ((real_pred_ratings - train_gt_ratings) ** 2).mean()

        # valid_rmse = evaluate(params=params, net=net, dataset=dataset, segment='valid')
        valid_mse = evaluate(params=params, net=net, dataset=dataset, segment='valid')
        # logging_str = f"Iter={iter_idx:>4d}, Train_RMSE={train_rmse:.4f}, ED_MI={ed_loss:.4f}, ND_MI={nd_loss:.4f}, " \
        #               f"Valid_RMSE={valid_rmse:.4f}, "
        logging_str = f"Iter={iter_idx:>4d}, Train_MSE={train_mse:.4f}, ED_MI={ed_loss:.4f}, ND_MI={nd_loss:.4f}, " \
                      f"Valid_MSE={valid_mse:.4f}, "

        # if valid_rmse < best_valid_rmse:
        #     best_valid_rmse = valid_rmse
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            no_better_valid = 0
            best_iter = iter_idx
            # test_rmse = evaluate(params=params, net=net, dataset=dataset, segment='test')
            # best_test_rmse = test_rmse
            test_mse = evaluate(params=params, net=net, dataset=dataset, segment='test')
            best_test_mse = test_mse

            # logging_str += ' Test RMSE={:.4f}'.format(test_rmse)
            logging_str += ' Test MSE={:.4f}'.format(test_mse)

        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
                print("Early stopping threshold reached. Stop training.")
                logger.info("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor,
                             params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    print("\tChange the LR to %g" % new_lr)
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        print(logging_str)
        # logger.info(logging_str)

    # print(f'Best Iter Idx={best_iter:>4d}, Best Valid RMSE={best_valid_rmse:.4f}, Best Test RMSE={best_test_rmse:.4f}')
    logger.info(logging_str)
    print(f'Best Iter Idx={best_iter:>4d}, Best Valid MSE={best_valid_mse:.4f}, Best Test MSE={best_test_mse:.4f}')
    logger.info(f'Best Iter Idx={best_iter:>4d}, Best Valid MSE={best_valid_mse:.4f}, Best Test MSE={best_test_mse:.4f}')
    # print(params.model_save_path)
    # logger.info(params.logger_file)
    return best_test_mse


if __name__ == '__main__':
    config_args = config()
    

    logger = request_logger(config_args.logger, config_args)
    logger.info(":::::::datetime:::::::{}:::::::::::::::::::::::::::::::::::".format(datetime.now().strftime('%Y-%m-%d%H:%M:%S')))
    
    best_lst = []
    for i in range(5):
        best_test = train(config_args)
        best_lst.append(best_test)
    best_lst.append(sum(best_lst)/5)
    with open(ROOT+'log_final/'+'best.csv', 'a', newline='')  as f:
        w = csv.writer(f)
        w.writerow(best_lst)