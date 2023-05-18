import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import os, sys, time
import math
import random
import numpy as np


def decov(x, y, diag=False):
    b = x.size(0)
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    x = x - x_mean[None, :]
    y = y - y_mean[None, :]
    mat = (x.t()).mm(y) / b

    decov_loss = 0.5 * torch.norm(mat, p='fro')**2
    if diag:
        decov_loss = decov_loss - 0.5 * torch.norm(torch.diag(mat))**2

    return decov_loss


class TimeAttn(nn.Module):
    def __init__(self, dim, time_dim, beta):
        super(TimeAttn, self).__init__()
        self.beta = beta
        self.temperature = np.sqrt(dim * 1.0)

        # self.pos_emb = nn.Embedding(150, time_dim)
        # self.rel_emb = nn.Embedding(150, time_dim)
        self.pos_emb = nn.Embedding(512, time_dim)
        self.rel_emb = nn.Embedding(512, time_dim)

        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_rp = nn.Linear(2 * time_dim, 1, bias=False)

        self.rate = nn.Parameter(torch.ones(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.pos_emb.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.rel_emb.weight, -0.1, 0.1)

        torch.nn.init.uniform_(self.fc_k.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.fc_q.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.fc_rp.weight, -0.1, 0.1)

    def forward(self, out, hn, pos_ind, rel_dt, abs_dt):
        pad_mask = pos_ind == 0

        pos_ind = self.pos_emb(pos_ind)
        rel_dt = self.rel_emb(rel_dt)

        attn_k = self.fc_k(out)  #(b, s, d)
        attn_q = self.fc_q(hn)  #(b,d)

        attn_0 = (torch.bmm(attn_k, attn_q.unsqueeze(-1)).squeeze(-1)
                  ) / self.temperature  #(b, s)
        attn_1 = self.fc_rp(torch.cat([rel_dt, pos_ind],
                                      -1)).squeeze(-1)  #(b, s)
        attn = attn_0 + self.beta * attn_1

        attn = attn.masked_fill(pad_mask, -np.inf)
        attn = F.softmax(attn, 1)
        outputs = attn

        # outputs = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
        return outputs


class gru_module(nn.Module):
    def __init__(self, input_dim, gru_dim, time_dim, beta):
        super(gru_module, self).__init__()
        self.gru = nn.GRU(input_dim, gru_dim, batch_first=True)
        self.attention = TimeAttn(gru_dim, time_dim, beta)

    def forward(self, inputs, length, pos_ind, rel_dt, abs_dt):
        # list, int, numarray, numarray, numarray
        # print('inputs', inputs.device)

        sorted_len, sorted_idx = length.sort(0, descending=True)  # sorted_len : tensor([17, 17, 17, 17, 17,  9,  6,  4,  4,  3,  3,  2,  2,  2,  2,  2] # [16]
        # print('length', length)
        # print('sorted_len', sorted_len)


        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(inputs) # sorted_idx : tensor([ 5,  8, 11,  0, 12, 15,  7, 14,  3, 10,  6, 13,  1,  9,  2,  4],
        sorted_inputs = inputs.gather(0, index_sorted_idx.long()) # [16,17,100]


        inputs_pack = pack_padded_sequence(
            sorted_inputs, sorted_len.cpu(), batch_first=True) ### .cpu() 추가
        # print('input_pack', len(inputs_pack))
        out, hn = self.gru(inputs_pack.cuda()) ### 
        hn = torch.squeeze(hn, 0)
        out, lens_unpacked = pad_packed_sequence(
            out, batch_first=True, total_length=inputs.size(1))

        _, ori_idx = sorted_idx.sort(0, descending=False)
        unsorted_idx = ori_idx.view(-1, 1).expand_as(hn)
        hn = hn.gather(0, unsorted_idx.long())
        unsorted_idx = ori_idx.view(-1, 1, 1).expand_as(out)
        out = out.gather(0, unsorted_idx.long())
        # print('out', out)
        # print('hn', hn)
        # print('pos_ind', pos_ind)
        # print('rel_dt', rel_dt)
        # print('abs_dt', abs_dt)

        out = self.attention(out, hn, pos_ind, rel_dt, abs_dt)
        return out


