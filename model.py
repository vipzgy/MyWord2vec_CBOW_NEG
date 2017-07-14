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
        loss = []

        for idx, pi in enumerate(pos_u):
            j = self.u_embedding(Variable(torch.LongTensor(pi)))
            combine = torch.t(torch.sum(torch.t(j), 1))
            if idx == 0:
                emb_u = combine
            else:
                emb_u = torch.cat((emb_u, combine), 0)
        emb_v = self.v_embedding(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, 1)
        score = F.logsigmoid(score)
        loss.append(sum(score))

        for idx, ni in enumerate(neg_u):
            j = self.u_embedding(Variable(torch.LongTensor(ni)))
            combine = torch.t(torch.sum(torch.t(j), 1))
            if idx == 0:
                neg_emb_u = combine
            else:
                neg_emb_u = torch.cat((neg_emb_u, combine), 0)

        neg_emb_v = self.v_embedding(Variable(torch.from_numpy(numpy.array(neg_v)).type(torch.LongTensor)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, 1)
        neg_score = F.logsigmoid(-1 * neg_score)
        loss.append(sum(neg_score))

        return -1 * sum(loss)

    def save_embedding(self, id2word, file_name):
        # 输出的是u的
        embedding = self.v_embedding.weight.data.numpy()
        output = open(os.path.join(self.args.dir, file_name), 'w', encoding='utf-8')
        output.write('%d %d\n' % (len(id2word), self.args.emb_dim))
        output.flush()
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            output.write('%s %s\n' % (w, e))
            output.flush()
        output.close()
