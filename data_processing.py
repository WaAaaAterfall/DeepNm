import numpy as np
import pandas as pd
from utils import create_folder, load_nanodata, data_util

np.random.seed(323)

data_name = 'Gm'
data_dir = './data/'
target_dir = './processed/'
target_dir = target_dir + data_name + '/'
create_folder(target_dir)
pos_path = data_dir + data_name + '_pos.csv'
neg_path = data_dir + data_name + '_neg_random1.csv'

# data_util(pos_path)

train_nano, train_seq, val_nano, val_seq, test_nano, test_seq, y_train, y_val, y_test = load_nanodata(pos_path, neg_path)
print(y_train)
np.save(target_dir + 'train_nano.npy', train_nano)
np.save(target_dir + 'train_seq.npy', train_seq)
np.save(target_dir + 'val_nano.npy', val_nano)
np.save(target_dir + 'val_seq.npy', val_seq)
np.save(target_dir + 'test_nano.npy', test_nano)
np.save(target_dir + 'test_seq.npy', test_seq)
np.save(target_dir + 'y_train.npy', y_train)
np.save(target_dir + 'y_val.npy', y_val)
np.save(target_dir + 'y_test.npy', y_test)


