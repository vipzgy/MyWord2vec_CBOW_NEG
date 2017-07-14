# -*- coding: utf-8 -*-
import mytrain
from mydata import InputData
from model import SkipGramModel

import argparse

parser = argparse.ArgumentParser(description='Word2vec')
parser.add_argument('-lr', type=float, default=0.025)
parser.add_argument('-epochs', type=int, default=5)
parser.add_argument('-window-size', type=int, default=5)
parser.add_argument('-min-count', type=int, default=5)
parser.add_argument('-neg-count', type=int, default=5)
parser.add_argument('-batch-size', type=int, default=100)
parser.add_argument('-emb-dim', type=int, default=100)
parser.add_argument('-using-hs', action='store_true', default=False)

parser.add_argument('-dir', type=str, default='./data')
parser.add_argument('-no-cuda', action='store_true')
parser.add_argument('-test', action='store_true', default=False)
args = parser.parse_args()


if __name__ == '__main__':
    # data
    data = InputData('zhihu.txt', args)
    args.output_file_name = 'result.txt'

    # update args
    args.emb_size = len(data.word2id)

    # do
    skip_gram_model = SkipGramModel(args)
    mytrain.train(data, skip_gram_model, args)




