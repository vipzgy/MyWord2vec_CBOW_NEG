# -*- coding: utf-8 -*-
import os
import numpy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, args):
        super(SkipGramModel, self).__init__()

        self.args = args
        self.u_embedding = nn.Embedding(args.emb_size, args.emb_dim, sparse=True)
        self.v_embedding = nn.Embedding(args.emb_size, args.emb_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.args.emb_dim
        self.u_embedding.weight.data.uniform_(-initrange, initrange)
        self.v_embedding.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_u = []
        for i in range(len(pos_u)):
            emb_v_v = self.u_embedding(Variable(torch.LongTensor(pos_u[i])))
            emb_v_v_numpy = emb_v_v.data.numpy()
            emb_v_v_numpy = numpy.sum(emb_v_v_numpy, axis=0)
            emb_v_v_list = emb_v_v_numpy.tolist()
            emb_u.append(emb_v_v_list)
        emb_u = Variable(torch.FloatTensor(emb_u))
        emb_v = self.u_embedding(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))

        neg_emb_u = []
        for i in range(len(neg_u)):
            neg_emb_v_v = self.u_embedding(Variable(torch.LongTensor(neg_u[i])))
            neg_emb_v_v_numpy = neg_emb_v_v.data.numpy()
            neg_emb_v_v_numpy = numpy.sum(neg_emb_v_v_numpy, axis=0)
            neg_emb_v_v_list = neg_emb_v_v_numpy.tolist()
            neg_emb_u.append(neg_emb_v_v_list)
        neg_emb_u = Variable(torch.FloatTensor(neg_emb_u))

        neg_emb_v = self.u_embedding(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)


    def save_embedding(self, id2word, file_name):
        # 输出的是v的
        embedding = self.v_embedding.weight.data.numpy()
        output = open(os.path.join(self.args.dir, 'v'), 'w', encoding='utf-8')
        output.write('%d %d\n' % (len(id2word), self.args.emb_dim))
        output.flush()
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            output.write('%s %s\n' % (w, e))
            output.flush()
        output.close()

        embedding = self.u_embedding.weight.data.numpy()
        output = open(os.path.join(self.args.dir, 'cbow-neg1'), 'w', encoding='utf-8')
        output.write('%d %d\n' % (len(id2word), self.args.emb_dim))
        output.flush()
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            output.write('%s %s\n' % (w, e))
            output.flush()
        output.close()
