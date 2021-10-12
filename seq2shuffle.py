import numpy as np
from utils import create_folder
from exp_utils import dinuc_shuffle

np.random.seed(323)

data_name = 'Tm'
dataset_name = 'test'
data_dir = './processed/' + data_name + '/'
target_dir = data_dir + 'shuffled/'
create_folder(target_dir)

seq_data = np.load(data_dir + dataset_name + '_seq.npy', allow_pickle=True)
y_data = np.load(data_dir + 'y_' + dataset_name + '.npy', allow_pickle=True)

pos_seq = seq_data[np.where(y_data == 1)[0]]
print(pos_seq.shape)

shuffle_times = 20

print(dinuc_shuffle(pos_seq[0]).shape)

shuffle_seqs = []
for i in np.arange(len(pos_seq)):
    shuffle_refs = []
    for j in np.arange(shuffle_times):
        org = pos_seq[i]
        shuffle_refs.append(dinuc_shuffle(org)[np.newaxis, ...])
    shuffle_seqs.append(np.concatenate(shuffle_refs, axis=0))

np.save(target_dir + dataset_name + '_shuffled_ref.npy', shuffle_seqs)



