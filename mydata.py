# -*- coding: utf-8 -*-
import os
import numpy
from collections import deque

numpy.random.seed(12345)


class InputData:
    def __init__(self, file_name, args):
        self.args = args
        # 队列存储所有配对
        self.word_pair_catch = deque()
        # 采样表
        self.sample_table = []
        # 去掉频率低于mini_count后所有的单词
        self.sentence_length = 0
        # 句子个数
        self.sentence_count = 0
        # 词 --> id
        self.word2id = {}
        # id --> 词
        self.id2word = {}
        # 词频率
        self.word_frequency = {}
        # 去重 去低频次 之后单词个数
        self.word_count = 0
        self.input_file = open(os.path.join(self.args.dir, file_name), encoding='utf-8').readlines()

        self.get_words()
        self.init_sample_table()

        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        print('Sentence count: %d' % (self.sentence_count))

    # 输入，统计所有词
    def get_words(self):
        word_frequency = {}
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        wid = 0
        for w, c in word_frequency.items():
            if c < self.args.min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        sample_table_size = 1e8
        # print(list(self.word_frequency.values()).__class__, self.word_frequency.values().__class__)
        pow_frequency = numpy.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

# vip 生成pairs，u代表content（w），v代表正采样w
    def get_batch_pairs(self):
        while len(self.word_pair_catch) < self.args.batch_size:
            for sentence in self.input_file:
                if sentence is None or sentence == '':
                    continue
                word_ids = []
                for word in sentence.strip().split(' '):
                    try:
                        word_ids.append(self.word2id[word])
                    except:
                        continue
                for i, u in enumerate(word_ids):
                    contentw = []
                    for j, v in enumerate(word_ids):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        elif j >= max(0, i - self.args.window_size + 1) and j <= min(len(word_ids), i + self.args.window_size-1):
                            contentw.append(v)
                    if len(contentw) == 0:
                        continue
                    self.word_pair_catch.append((contentw, u))
        batch_pairs = []
        for _ in range(self.args.batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

# vip ???negative sampling
    def get_pairs_by_neg_sampling(self, pos_word_pair):
        neg_word_pair = []

        for pair in pos_word_pair:
            neg_v = numpy.random.choice(self.sample_table, size=self.args.neg_count)
            neg_word_pair += zip([pair[0]] * self.args.neg_count, neg_v)
        return pos_word_pair, neg_word_pair

# doubt
    def evaluate_pair_count(self):
        return self.sentence_length * (2 * self.args.window_size - 1) - (self.sentence_count - 1) * (
        1 + self.args.window_size) * self.args.window_size











