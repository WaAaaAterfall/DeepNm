import numpy as np

data_name = 'Am'
data_dir = '/home/daiyun/DATA/Nano2pO/processed/'
data_dir = data_dir + data_name + '/'
target_path = data_dir + data_name + '.txt'

test_seq = np.load(data_dir + 'test_seq.npy', allow_pickle=True)
test_seq = np.argmax(test_seq, axis=-1)
tmp = np.array(['A', 'C', 'G', 'T'])
tmp = [''.join(seq) for seq in tmp[test_seq]]

with open(target_path, 'w') as txt_file:
    for idx, seq in enumerate(tmp):
        txt_file.write('>test' + str(idx+1) + '\n')
        txt_file.write(seq + '\n')


