# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch.optim as optim

def train(data, model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    pair_count = data.evaluate_pair_count()
    batch_count = args.epochs * pair_count / args.batch_size
    process_bar = tqdm(range(int(batch_count)))
    model.save_embedding(data.id2word, 'begin_embedding.txt')

    for i in process_bar:
        pos_pairs = data.get_batch_pairs()

        pos_pairs, neg_pairs = data.get_pairs_by_neg_sampling(pos_pairs)

        pos_u = [pair[0] for pair in pos_pairs]
        pos_v = [pair[1] for pair in pos_pairs]
        neg_u = [pair[0] for pair in neg_pairs]
        neg_v = [pair[1] for pair in neg_pairs]

        optimizer.zero_grad()
        loss = model.forward(pos_u, pos_v, neg_u, neg_v)
        loss.backward()
        optimizer.step()

        process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
            (loss.data[0], optimizer.param_groups[0]['lr']))

        if i * args.batch_size % 100000 == 0:
            lr = args.lr * (1.0 - 1.0 * i / batch_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    model.save_embedding(data.id2word, args.output_file_name)