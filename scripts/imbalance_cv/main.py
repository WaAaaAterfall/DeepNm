import numpy as np
# from training import train_diff_model
from training import train_diff_model
from testing import eval_model
from utils import create_folder, merge_cv_results
from config import Config

c = Config()

training = False
c.nfold = 5
c.nrep = 10

c.data_name = 'Cm'
c.nano = True
c.coverage_only = False
c.no_quality = False
c.model_funname = 'Nano2pO'
c.data_dir = '../../processed/' + c.data_name + '/imbalance_cv/'
c.target_dir = c.data_dir + 'cp_dir/'
create_folder(c.target_dir)
if c.coverage_only:
    cp_dir = c.target_dir + c.model_funname.replace('create_', '') + '_c/'
else:
    cp_dir = c.target_dir + c.model_funname.replace('create_', '') + '/'

if __name__ == '__main__':
    if training == True:
        for i in np.arange(1, c.nfold + 1):
            c.cp_path = cp_dir + 'f' + str(i) + '_t1.ckpt'
            c.valid_idx = i
            c.train_idx = list(range(1, c.nfold + 1))
            c.train_idx.remove(c.valid_idx)
            train_diff_model(c)
    else:
        valid_results = []
        for i in np.arange(1, c.nfold + 1):
            c.cp_path = cp_dir + 'f' + str(i) + '_t1.ckpt'
            c.valid_idx = i
            # c.data_name = 'fold' + str(c.valid_idx)
            results, label, pred= eval_model(c)
            valid_results.append(results[0])
        valid_results = merge_cv_results(valid_results)
