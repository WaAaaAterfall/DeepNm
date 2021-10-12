import numpy as np

def cross_val(fold_num, r):
    data_dir = "C:\\Users\\chels\\Desktop\\Nano2pO_update\\processed\\"
    dire = "C:\\Users\\chels\\Desktop\\Nano2pO_update\\data\\"
    cul_loss = 0
    iter = 0
    data_pos = np.load(dire + r +'_pos.npy', allow_pickle=True)
    data_neg = np.load(dire + r+ '_neg.npy', allow_pickle=True)
    for iter in range(fold_num):
        y_test = np.load(data_dir + r + '\\imbalance_cv\\fold'+str(iter+1)+'_label.npy', allow_pickle=True)
        nano_test = np.load(data_dir + r + '\\imbalance_cv\\fold'+str(iter+1)+'_nano.npy', allow_pickle=True)
        seq_test = np.load(data_dir + r + '\\imbalance_cv\\fold'+str(iter+1)+'_seq.npy', allow_pickle=True)
        idx_test = np.load(data_dir + r + '\\imbalance_cv\\fold'+str(iter+1)+'_seq.npy', allow_pickle=True)
        train_pos = np.delete(data_pos, idx_test.item()['pos'+str(iter+1)], axis=0)
        train_neg = np.delete(data_neg, idx_test.item()['neg'+str(iter+1)], axis=0)
        train = np.concatenate([train_pos, train_neg], axis=0)
        nano_train = train[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
        seq_train = train[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
        y_train = np.concatenate([np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])])

