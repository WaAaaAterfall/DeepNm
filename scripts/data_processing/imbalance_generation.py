import os
import time
import numpy as np
import pandas as pd
#import rpy2.robjects as ro
from sklearn.model_selection import train_test_split

#readRDS = ro.r['readRDS']

data_dir = '../../data/'
rm = ['Am', 'Cm', 'Gm', 'Tm']

count = 0
for m in rm:
    count = count+1
    print(m+str(count))
    pos_data = np.load(data_dir + m + '_pos.npy', allow_pickle=True)
    pos_nano = pos_data[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
    pos_seq = pos_data[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
    neg_data = np.load(data_dir + m + '_neg_15.npy', allow_pickle=True)
    neg_nano = neg_data[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
    neg_seq = neg_data[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
    nano_data = np.concatenate([pos_nano, neg_nano], axis=0)
    print(np.isnan(nano_data).any())

    # normalize nanopore derived features
    nano_min = np.min(nano_data, axis=(0, 1))
    nano_max = np.max(nano_data, axis=(0, 1))
    nano_data = (nano_data - nano_min) / (nano_max - nano_min)

    seq_data = np.concatenate([pos_seq, neg_seq], axis=0)
    print(np.isnan(seq_data).any())
    y_label = np.concatenate([np.ones(pos_nano.shape[0]).reshape(-1, 1),
                              np.zeros(neg_nano.shape[0]).reshape(-1, 1)], axis=0)
    target_dir = '../../processed/' + m + '/imbalance15/'
    train_seq, test_seq, train_nano, test_nano, y_train, y_test = train_test_split(
        seq_data, nano_data, y_label, test_size=0.2, random_state=323
    )
    print(train_seq.shape)
    print(test_seq.shape)
    print(train_nano.shape)
    np.save(target_dir + 'train_seq.npy', train_seq)
    np.save(target_dir + 'test_seq.npy', test_seq)
    np.save(target_dir + 'train_nano.npy', train_nano)
    np.save(target_dir + 'test_nano.npy', test_nano)
    np.save(target_dir + 'y_train.npy', y_train)
    np.save(target_dir + 'y_test.npy', y_test)

# start_time = time.time()
# pos_data = pd.read_csv(data_dir + 'Am_pos.csv', index_col=0)
# pos_data.iloc[:, :123] = pos_data.iloc[:, :123].replace(np.nan, 0)
# pos_data.iloc[:, 123:164] = pos_data.iloc[:, 123:164].replace(np.nan, -1)
# print(pos_data.iloc[:, :123])
# print(pos_data.iloc[:, 123:164])
# Am_neg = np.load(data_dir + 'Am_neg.npy', allow_pickle=True)
# print(round(time.time() - start_time, 2))
# np.save(data_dir + 'Am_pos.npy', pos_data.values)
