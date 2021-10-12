import os
import numpy as np
import pandas as pd

data_dir = '../../data/'
rm = ['Am', 'Cm', 'Gm', 'Tm']

nfold = 5

for m in rm:
    pos_data = np.load(data_dir + m + '_pos.npy', allow_pickle=True)
    print(np.isnan(pos_data).any())
    pos_nano = pos_data[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
    pos_seq = pos_data[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
    print(pos_seq.shape)
    neg_data = np.load(data_dir + m + '_neg.npy', allow_pickle=True)
    print(neg_data.shape)
    print(int(neg_data.shape[0]/2))
    idx_part = np.random.permutation(neg_data.shape[0])[:int(neg_data.shape[0]/2)]
    neg_data_select = neg_data[idx_part]
    print(neg_data_select.shape)
    np.save(data_dir + m + '_neg_15.npy', neg_data_select)
