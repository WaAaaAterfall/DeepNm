import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from nets import *

tfk = tf.keras
tfko = tfk.optimizers
tfkm = tfk.metrics
tfkc = tfk.callbacks
tfdd = tf.data.Dataset


def create_folder(dir):
    if not os.path.exists(dir):
        print('Create folder: ', dir)
        os.makedirs(dir)


def data_util(path):
    # Read data into numpy arrays
    data = pd.read_csv(path)
    # Random sampling for neg data
    # if 'neg' in path:
    # 	data = data.sample(frac=0.1, replace=False, axis=0)
    print('Finished reading the Dataset ({} samples found, each dim = {})'.format(len(data), data.shape[1]))

    # Check missing value for one-hot code
    for i in reversed(range(data.iloc[:, 165:].shape[0])):
        if data.iloc[i, 165:].isnull().values.any():
            print('missing value in sample ', (i + 1))
            data = data.drop(labels=[i])
    if not data.iloc[:, 165:].isnull().values.any():
        print('Finish filtering')

    data = np.array(data)[:, 1:].astype(np.float32)
    if 'pos' in path:
        label = np.ones((data.shape[0], 1))
    else:
        label = np.zeros((data.shape[0], 1))
    print('After writing in np.array, data shape: ', data.shape)

    indice_isnan = np.isnan(data)
    data[indice_isnan] = 0
    print('Any NA in data:', True in np.isnan(data))

    tmp = data[:, :164].reshape(-1, 4, 41).transpose((0, 2, 1))
    nano_min = np.repeat(np.min(tmp, axis=(0, 1)), 41)
    nano_max = np.repeat(np.max(tmp, axis=(0, 1)), 41)
    data[:, :164] = (data[:, :164] - nano_min) / (nano_max - nano_min)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    X_train_nano = X_train[:, :164]
    X_train_seq = X_train[:, 164:]
    X_val_nano = X_val[:, :164]
    X_val_seq = X_val[:, 164:]
    X_test_nano = X_test[:, :164]
    X_test_seq = X_test[:, 164:]

    return X_train_nano, X_train_seq, X_val_nano, X_val_seq,\
           X_test_nano, X_test_seq, y_train, y_val, y_test


def load_nanodata(pos_path, neg_path):
    # lode data
    X_pos_train_nano, X_pos_train_seq, X_pos_val_nano, X_pos_val_seq, X_pos_test_nano, X_pos_test_seq, \
    y_pos_train, y_pos_val, y_pos_test = data_util(pos_path)
    X_neg_train_nano, X_neg_train_seq, X_neg_val_nano, X_neg_val_seq, X_neg_test_nano, X_neg_test_seq, \
    y_neg_train, y_neg_val, y_neg_test = data_util(neg_path)

    # merge pos and neg data
    X_train_nano = np.append(X_pos_train_nano, X_neg_train_nano, axis=0)
    X_train_nano = np.reshape(X_train_nano, (X_train_nano.shape[0], 4, 41)).transpose((0, 2, 1))
    X_train_seq = np.append(X_pos_train_seq, X_neg_train_seq, axis=0)
    X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], 1001, 4))
    X_val_nano = np.append(X_pos_val_nano, X_neg_val_nano, axis=0)
    X_val_nano = np.reshape(X_val_nano, (X_val_nano.shape[0], 4, 41)).transpose((0, 2, 1))
    X_val_seq = np.append(X_pos_val_seq, X_neg_val_seq, axis=0)
    X_val_seq = np.reshape(X_val_seq, (X_val_seq.shape[0], 1001, 4))
    X_test_nano = np.append(X_pos_test_nano, X_neg_test_nano, axis=0)
    X_test_nano = np.reshape(X_test_nano, (X_test_nano.shape[0], 4, 41)).transpose((0, 2, 1))
    X_test_seq = np.append(X_pos_test_seq, X_neg_test_seq, axis=0)
    X_test_seq = np.reshape(X_test_seq, (X_test_seq.shape[0], 1001, 4))
    y_train = np.append(y_pos_train, y_neg_train, axis=0)
    y_val = np.append(y_pos_val, y_neg_val, axis=0)
    y_test = np.append(y_pos_test, y_neg_test, axis=0)

    train_shuffle_idx = np.random.permutation(len(y_train))
    X_train_nano = X_train_nano[train_shuffle_idx]
    X_train_seq = X_train_seq[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    return X_train_nano, X_train_seq, X_val_nano, X_val_seq, X_test_nano, X_test_seq, y_train, y_val, y_test


def linearly_interpolate(sample, reference=False, num_steps=20):
    if reference != None:
        pass
    else:
        reference = np.zeros(sample.shape)

    assert sample.shape == reference.shape

    ret = np.zeros(tuple([num_steps+1] + [i for i in sample.shape]))

    for s in range(num_steps+1):
        ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

    return ret.astype(np.float32), num_steps, (sample - reference)


@tf.function
def obtain_gradients(model, inputs, if_feature=False):
    if not if_feature:
        with tf.GradientTape() as Tape:
            x1 = tf.convert_to_tensor(inputs[0])
            x2 = tf.convert_to_tensor(inputs[1])
            Tape.watch(x1)
            output_probs = model([x1, x2], training=False)
        dp_dx = Tape.gradient(output_probs, x1)
    else:
        with tf.GradientTape() as Tape:
            x1 = tf.convert_to_tensor(inputs[0])
            x2 = tf.convert_to_tensor(inputs[1])
            Tape.watch(x2)
            output_probs, _ = model([x1, x2], training=False)
        dp_dx = Tape.gradient(output_probs, x2)
    return dp_dx


def fixed_ig(seq, nano, model, reference=False, ig_step=20):

    samples, numsteps, step_sizes = linearly_interpolate(seq, reference, ig_step)

    nano = np.broadcast_to(nano, [ig_step + 1, nano.shape[0], nano.shape[1]])

    gradients = obtain_gradients(model, [samples, nano], if_feature=False)

    gradients = (gradients[:-1] + gradients[1:]) / 2
    hyp_scores = np.mean(gradients, axis=0)
    hyp_scores = hyp_scores - np.mean(hyp_scores, axis=-1)[..., np.newaxis]
    ig_scores = np.multiply(hyp_scores, step_sizes)
    return ig_scores, hyp_scores


def dishuffle_ig(seq, nano, model, reference, shuffle_times=20, ig_step=20):
    ig_scores_list = []
    hype_scores_list = []
    nano = np.broadcast_to(nano, [ig_step + 1, nano.shape[0], nano.shape[1]])
    for sidx in np.arange(shuffle_times):
        ref = reference[sidx]
        samples, numsteps, step_sizes = linearly_interpolate(seq, ref, ig_step)

        gradients = []
        for i in np.arange(ig_step + 1):
            gradients.append(obtain_gradients(model, [samples, nano], if_feature=False))

        gradients = np.concatenate(gradients, axis=0)
        gradients = (gradients[:-1] + gradients[1:]) / 2
        hyp_scores = np.mean(gradients, axis=0)
        hyp_scores = hyp_scores - np.mean(hyp_scores, axis=-1)[..., np.newaxis]
        ig_scores = np.multiply(hyp_scores, step_sizes)

        ig_scores_list.append(ig_scores[np.newaxis, ...])
        hype_scores_list.append(hyp_scores[np.newaxis, ...])

    mean_scores = np.mean(np.concatenate(ig_scores_list, axis=0), axis=0)
    mean_hype_scores = np.mean(np.concatenate(hype_scores_list, axis=0), axis=0)
    return mean_scores, mean_hype_scores