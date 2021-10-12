import numpy as np
from utils import train_diff_model, create_folder
from config import Config

c = Config()

c.training = True
c.data_name = 'Cm'
c.nano = True
c.coverage_only = False
c.no_quality = False
c.model_funname = 'Nano2pO'
c.data_dir = './processed/' + c.data_name + '/'
c.target_dir = c.data_dir + 'cp_dir/'
create_folder(c.target_dir)
c.cp_path = c.target_dir + c.model_funname.replace('create_', '') + '.ckpt'


if __name__ == '__main__':
    if c.training == True:
        train_diff_model(c)





