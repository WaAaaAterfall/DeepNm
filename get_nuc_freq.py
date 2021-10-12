import numpy as np

data_name = 'Tm'
data_dir = './processed/' + data_name + '/'

train_seq = np.load(data_dir + 'train_seq.npy', allow_pickle=True)

print(train_seq.mean(axis=(0, 1)))

# nuc_counts = np.concatenate(nuc_counts)
# nuc_freq = np.sum(nuc_counts, axis=0)
# nuc_freq = nuc_freq / np.sum(nuc_freq)
# print(nuc_freq)