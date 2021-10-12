import os
import time
import numpy as np
import pandas as pd
#import rpy2.robjects as ro

np.random.seed(323)

#readRDS = ro.r['readRDS']

data_dir = '../../data/'
rm = ['Am', 'Cm', 'Gm', 'Tm']

nfold = 5
for m in rm:
    pos_data = np.load(data_dir + m + '_pos.npy', allow_pickle=True)
    print(np.isnan(pos_data).any())
    pos_nano = pos_data[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
    pos_seq = pos_data[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
    print(pos_seq.shape)
    neg_data = np.load(data_dir + m + '_neg_15.npy', allow_pickle=True)
    neg_nano = neg_data[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
    print(neg_nano.shape)
    neg_seq = neg_data[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
    print(neg_seq.shape)
    nano_data = np.concatenate([pos_nano, neg_nano], axis=0)
    print(np.isnan(nano_data).any())

    # normalize nanopore derived features
    nano_min = np.min(nano_data, axis=(0, 1))
    print(nano_min)
    nano_max = np.max(nano_data, axis=(0, 1))
    print(nano_max)
    nano_data = (nano_data - nano_min) / (nano_max - nano_min)
    pos_nano = nano_data[:pos_nano.shape[0]]
    neg_nano = nano_data[pos_nano.shape[0]:]

    seq_data = np.concatenate([pos_seq, neg_seq], axis=0)
    print(np.isnan(seq_data).any())
    y_label = np.concatenate([np.ones(pos_nano.shape[0]).reshape(-1, 1),
                              np.zeros(neg_nano.shape[0]).reshape(-1, 1)], axis=0)
    target_dir = '../../processed/' + m + '/imbalance_cv15/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pos_sidx = np.random.permutation(pos_nano.shape[0])
    neg_sidx = np.random.permutation(neg_nano.shape[0])
    npos = int(0.2 * len(pos_sidx))
    nneg = int(0.2 * len(neg_sidx))
    fold_split_idx = {}
    for i in np.arange(1, nfold+1):
        if i != nfold:
            fold_split_idx['pos' + str(i)] = pos_sidx[(i-1)*npos:i*npos]
            fold_split_idx['neg' + str(i)] = neg_sidx[(i-1)*nneg:i*nneg]
        else:
            fold_split_idx['pos' + str(i)] = pos_sidx[(i-1)*npos:]
            fold_split_idx['neg' + str(i)] = neg_sidx[(i-1)*nneg:]
        fold_nano = np.concatenate([pos_nano[fold_split_idx['pos' + str(i)]],
                                    neg_nano[fold_split_idx['neg' + str(i)]]])
        fold_seq = np.concatenate([pos_seq[fold_split_idx['pos' + str(i)]],
                                   neg_seq[fold_split_idx['neg' + str(i)]]])
        fold_label = np.concatenate([np.ones(len(fold_split_idx['pos' + str(i)])).reshape(-1, 1),
                                     np.zeros(len(fold_split_idx['neg' + str(i)])).reshape(-1, 1)]).astype(np.int32)
        fold_sidx = np.random.permutation(fold_nano.shape[0])
        fold_nano = fold_nano[fold_sidx]
        fold_seq = fold_seq[fold_sidx]
        fold_label = fold_label[fold_sidx]
        np.save(target_dir + 'fold' + str(i) + '_nano.npy', fold_nano)
        np.save(target_dir + 'fold' + str(i) + '_seq.npy', fold_seq)
        np.save(target_dir + 'fold' + str(i) + '_label.npy', fold_label)
    np.save(target_dir + 'fold_split_idx.npy', fold_split_idx)
