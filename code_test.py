import numpy as np
from nets import Nano2pO, ResBlock

data_dir = './processed/Am/'
test_nano = np.load(data_dir + 'test_nano.npy', allow_pickle=True)
print(np.max(test_nano, axis=(0, 1)))
print(np.quantile(test_nano, axis=(0, 1), q=0.95))


# model = Nano2pO()
# data = np.ones([1, 1001, 4]).astype(np.float32)
#
# model([data, data])
# model.summary()