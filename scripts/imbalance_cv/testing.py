import time
import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import *
from utils import create_folder
from prettytable import PrettyTable

tfk = tf.keras
tfdd = tf.data.Dataset

def eval_model(c):

    # dataset = 'valid_'
    print('Loading data!')
    valid_out = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_label.npy', allow_pickle=True)
    valid_out = valid_out.astype(np.int32).reshape(-1, 1)
    valid_seq = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_seq.npy', allow_pickle=True)
    valid_seq = valid_seq[:, 355:646, :].astype(np.float32)

    if c.nano:
        valid_nano = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_nano.npy',
                             allow_pickle=True)

        if c.coverage_only:
            valid_nano = valid_nano[..., 1:2]

        if c.no_quality:
            valid_nano = valid_nano[..., :-1]

        valid_dataset = tfdd.from_tensor_slices((valid_seq, valid_nano))
        valid_dataset = valid_dataset.batch(128)
    else:
        valid_dataset = tfdd.from_tensor_slices(valid_seq)
        valid_dataset = valid_dataset.batch(256)

    print('Creating model')
    if isinstance(c.model_funname, str):
        dispatcher = {'create_model1': create_model1,
                      'create_model2': create_model2,
                      'create_model3': create_model3,
                      'create_model4': create_model4,
                      'DeepOMe': DeepOMe,
                      'Nano2pO': Nano2pO,
                      'Seq2pO': Seq2pO,
                      'NanoOnly': NanoOnly}
        try:
            model_funname = dispatcher[c.model_funname]
        except KeyError:
            raise ValueError('invalid input')

    model = model_funname()
    model.load_weights(c.cp_path)

    results = []
    pred = []
    for tdata in valid_dataset:
        p = model(tdata, training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0)

    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

    thres = 0.5
    accuracy_scores.append(accuracy_score(y_true=valid_out, y_pred=pred > thres))
    f1_scores.append(f1_score(y_true=valid_out, y_pred=pred > thres))
    recall_scores.append(recall_score(y_true=valid_out, y_pred=pred > thres))
    precision_scores.append(precision_score(y_true=valid_out, y_pred=pred > thres))
    MCCs.append(matthews_corrcoef(y_true=valid_out, y_pred=pred > thres))
    auROCs.append(roc_auc_score(y_true=valid_out, y_score=pred))
    auPRCs.append(average_precision_score(y_true=valid_out, y_score=pred))

    if c.nfold:
        results.append(np.array([accuracy_scores[0], f1_scores[0],
                                 recall_scores[0], precision_scores[0],
                                 MCCs[0], auROCs[0], auPRCs[0]]).reshape(1, -1))
    else:
        table = PrettyTable()
        column_names = ['Accuracy', 'recall', 'precision', 'f1', 'MCC', 'auROC', 'auPRC']
        table.add_column(column_names[0], np.round(accuracy_scores, 4))
        table.add_column(column_names[1], np.round(recall_scores, 4))
        table.add_column(column_names[2], np.round(precision_scores, 4))
        table.add_column(column_names[3], np.round(f1_scores, 4))
        table.add_column(column_names[4], np.round(MCCs, 4))
        table.add_column(column_names[5], np.round(auROCs, 4))
        table.add_column(column_names[6], np.round(auPRCs, 4))

        print(table)

    if c.nfold:
        # np.save(c.save_dir + 'fold' + str(c.test_idx) + '.npy', pred)
        return results, valid_out, pred
    else:
        return None