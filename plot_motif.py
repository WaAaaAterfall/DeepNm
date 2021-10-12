import numpy as np
from visualization import plot_weights
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

data_name = 'Am'
data_dir = './results/' + data_name + '/'
motif_name = data_name + '_motif2'
motif = np.load(data_dir + motif_name + '.npy', allow_pickle=True)
# motif = motif[:7, :] # Am - motif2
plot_weights(motif, figsize=(motif.shape[0], 2))
plt.savefig(data_dir + motif_name + '.pdf', dpi=350, bbox_inches='tight', pad_inches=None)
# plt.show()