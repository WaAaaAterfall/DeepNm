import numpy as np
from training import train_diff_model, train_cross_val
from utils import create_folder
from config import Config
import argparse

c = Config()

c.training = True
c.data_name = 'Gm'
c.nano = True
c.coverage_only = False
c.no_quality = False
c.model_funname = 'Seq2pO'
c.data_dir = '../../processed/' + c.data_name + '/imbalance/'
c.cv_dir = '../../processed/' + c.data_name + '/imbalance_cv/'
c.target_dir = c.data_dir + 'cp_dir/'
create_folder(c.target_dir)
c.cp_path = c.target_dir + c.model_funname.replace('create_', '') + '.ckpt'


if __name__ == '__main__':
    if c.training == True:
        train_cross_val(c)
        #train_diff_model(c)





