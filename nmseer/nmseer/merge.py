import numpy as np
import pandas as pd
from utils import onehot_pred, knf_pred, psdsp_pred, merge_cv_results
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score

data_name = 'Tm'
data_dir = '../../processed/'
data_dir = data_dir + data_name + '/imbalance_cv/'
nfold = 5
onehot_results = onehot_pred(data_name)
knf_results = knf_pred(data_name)
psdsp_results = psdsp_pred(data_name)
eval_results = []

for idx in range(nfold):
    valid_idx = idx+1
    valid_label = np.load(data_dir + 'fold' + str(valid_idx) + '_label.npy', allow_pickle=True).reshape(-1)
    onehot_pred = onehot_results[idx]
    knf_pred = knf_results[idx]
    psdsp_pred = psdsp_results[idx]
    y_pred = 1/3*onehot_pred + 1/3*knf_pred + 1/3*psdsp_pred

    thres = 0.5
    acc = accuracy_score(y_true=valid_label, y_pred=y_pred > thres)
    f1 = f1_score(y_true=valid_label, y_pred=y_pred > thres)
    recall = recall_score(y_true=valid_label, y_pred=y_pred > thres)
    precision = precision_score(y_true=valid_label, y_pred=y_pred > thres)
    MCC = matthews_corrcoef(y_true=valid_label, y_pred=y_pred > thres)
    auc = roc_auc_score(y_true=valid_label, y_score=y_pred)
    ap = average_precision_score(y_true=valid_label, y_score=y_pred)

    if nfold > 1:
        eval_results.append(np.array([acc, f1, recall, precision, MCC, auc, ap]).reshape(1, -1))
        print('result of fold '+ str(valid_idx) + ' has been saved.')

merge_cv_results(eval_results)
