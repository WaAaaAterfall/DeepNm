import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from nets import create_model1, create_model3
from utils import fixed_ig, dishuffle_ig, create_folder
from visualization import plot_weights
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


data_name = 'Tm'
ref = 'fixed'
model_funname = 'create_model1'
data_dir = './processed/' + data_name + '/'
result_dir = './results/' + data_name + '/'
create_folder(result_dir)
cp_path = data_dir + 'cp_dir/' + model_funname.replace('create_', '') + '.ckpt'
if ref == 'dishuffle':
    reference = np.load(data_dir + 'shuffled/test_shuffled_ref.npy', allow_pickle=True)
else:
    reference = None

model = create_model1()
model.load_weights(cp_path)

test_seq = np.load(data_dir + 'test_seq.npy', allow_pickle=True)
test_nano = np.load(data_dir + 'test_nano.npy', allow_pickle=True)
y_test = np.load(data_dir + 'y_test.npy', allow_pickle=True)

pred = model([test_seq, test_nano], training=False)

print('AUROC score: ', roc_auc_score(y_test, pred))

tp_idx = [i[0] for i in sorted(enumerate(pred), key=lambda x:x[1], reverse=True)
          if (pred[i[0]] > 0.5) and (y_test[i[0]] == 1)]


ig_scores = []
hype_scores = []
one_hot_data = []

if ref == 'fixed':
    inds = tp_idx[:20]
    for idx, ind in enumerate(inds):
        ig_score, _ = fixed_ig(test_seq[ind], test_nano[ind], model, reference=reference)
        plot_weights(ig_score[475:526], highlight={'red': [[25, 26]]})
        plt.savefig(result_dir + str(idx+1) + '.pdf', bbox_inches='tight', pad_inches=0, dpi=350)
        # plt.show()
        # plt.close()
        print('{}/{} finished !'.format(idx + 1, len(inds)))
elif ref == 'dishuffle':
    for idx, ind in enumerate(tp_idx):
        ig_score, hype_score = dishuffle_ig(test_seq[ind], test_nano[ind], model,
                                            reference=reference[ind],
                                            shuffle_times=20, ig_step=20)
        # plot_weights(ig_score[450:551], highlight={})
        # plt.show()
        # plot_weights(hype_score[450:551], highlight={})
        # plt.show()

        ig_scores.append(ig_score[450:551][np.newaxis, ...])
        hype_scores.append(hype_score[450:551][np.newaxis, ...])
        one_hot_data.append(test_seq[ind][450:551][np.newaxis, ...])

        print('{}/{} finished !'.format(idx + 1, len(tp_idx)))
    ig_scores = np.concatenate(ig_scores, axis=0)
    hype_scores = np.concatenate(hype_scores, axis=0)
    one_hot_data = np.concatenate(one_hot_data, axis=0)
    print(one_hot_data.shape)
    np.save(result_dir + 'shuffle_task_to_scores.npy', ig_scores)
    np.save(result_dir + 'shuffle_task_to_hyp_scores.npy', hype_scores)
    np.save(result_dir + 'shuffle_onehot_data.npy', one_hot_data)
else:
    raise('Current your selected reference method is not supported!')

# for idx, ind in enumerate(tp_idx[:20]):
#     ig_score, _ = fixed_ig(test_seq[ind], test_nano[ind], model, reference=reference)
#     plot_weights(ig_score[450:551], highlight={'red': [[50, 51]]})
#
#     plt.show()
