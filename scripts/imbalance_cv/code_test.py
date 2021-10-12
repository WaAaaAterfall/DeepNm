import numpy as np

data_dir = '/home/daiyun/DATA/Nano2pO/processed/Am/imbalance_cv/fold1_seq.npy'

tmp = np.load(data_dir, allow_pickle=True)
print(tmp.shape)