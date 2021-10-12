import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import *

tfk = tf.keras
tfko = tfk.optimizers
tfkm = tfk.metrics
tfkc = tfk.callbacks
tfdd = tf.data.Dataset


def train_diff_model(config):
    c = config

    print('Loading data!')
    train_seq = np.load(c.data_dir + 'train_seq.npy',
                        allow_pickle=True).astype(np.float32)
    test_seq = np.load(c.data_dir + 'test_seq.npy',
                       allow_pickle=True).astype(np.float32)

    y_train = np.load(c.data_dir + 'y_train.npy',
                      allow_pickle=True).reshape(-1).astype(np.int32)
    y_test = np.load(c.data_dir + 'y_test.npy',
                     allow_pickle=True).reshape(-1).astype(np.int32)

    train_seq = train_seq[:, 355:646, :]
    test_seq = test_seq[:, 355:646, :]

    nrep = 10
    pidx = y_train == 1
    nidx = y_train == 0
    train_seq = np.concatenate([np.repeat(train_seq[pidx], nrep, axis=0), train_seq[nidx]])
    y_train = np.concatenate([np.ones(sum(pidx) * nrep), np.zeros(sum(nidx))])
    sidx = np.random.permutation(y_train.shape[0])
    train_seq = train_seq[sidx]
    y_train = y_train[sidx]

    if c.nano:
        train_nano = np.load(c.data_dir + 'train_nano.npy',
                             allow_pickle=True).astype(np.float32)
        test_nano = np.load(c.data_dir + 'test_nano.npy',
                            allow_pickle=True).astype(np.float32)

        # train_nano[train_nano == -1] = 0
        # test_nano[test_nano == -1] = 0

        if c.coverage_only:
            train_nano = train_nano[..., 1:2]
            test_nano = test_nano[..., 1:2]

        if c.no_quality:
            train_nano = train_nano[..., :-1]
            test_nano = test_nano[..., :-1]

        train_nano = np.concatenate([np.repeat(train_nano[pidx], nrep, axis=0), train_nano[nidx]])
        train_nano = train_nano[sidx]

    print(train_seq.shape)
    print(train_nano.shape)
    print(y_train.shape)
    print(np.mean(y_train))
    print(test_seq.shape)
    print(test_nano.shape)
    print(y_test.shape)
    print(np.mean(y_test))

    train_dataset = tfdd.from_tensor_slices((train_seq, train_nano, y_train))
    train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = tfdd.from_tensor_slices((test_seq, test_nano, y_test))
    test_dataset = test_dataset.batch(128)

    print('Creating model')
    if isinstance(c.model_funname, str):
        dispatcher={'create_model1': create_model1,
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

    adam = tfko.Adam(lr=c.lr_init, epsilon=1e-08, decay=c.lr_decay)
    train_loss = tfkm.Mean()
    valid_loss = tfkm.Mean()
    train_auc = tfkm.AUC()
    valid_auc = tfkm.AUC()

    @tf.function()
    def train_step(train_seq, train_nano, train_out):
        with tf.GradientTape() as tape:
            prob = model((train_seq, train_nano), training=True)
            loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
            total_loss = loss + tf.reduce_sum(model.losses)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            adam.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_auc(y_true=train_out, y_pred=prob)

    @tf.function()
    def valid_step(valid_seq, valid_nano, valid_out):
        prob = model((valid_seq, valid_nano), training=False)
        vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

        valid_loss(vloss)
        valid_auc(y_true=valid_out, y_pred=prob)

    EPOCHS = c.epoch
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, EPOCHS + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        train_auc.reset_states()
        valid_auc.reset_states()

        estime = time.time()
        for tdata in train_dataset:
            train_step(tdata[0], tdata[1], tdata[2])
        print(f'Training of epoch {epoch} finished! Time cost is {round(time.time() - estime, 2)}s')

        vstime = time.time()
        for vdata in test_dataset:
            valid_step(vdata[0], vdata[1], vdata[2])

        new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
        if new_valid_monitor < current_monitor:
            if c.cp_path:
                model.save_weights(c.cp_path)
                print('val_loss improved from {} to {}, saving model to {}'.
                      format(str(current_monitor), str(new_valid_monitor), c.cp_path))
            else:
                print('val_loss improved from {} to {}, saving closed'.
                      format(str(current_monitor), str(new_valid_monitor)))

            current_monitor = new_valid_monitor
            patient_count = 0
        else:
            print('val_loss did not improved from {}'.format(str(current_monitor)))
            patient_count += 1

        if patient_count == 10:
            break

        template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}"
        print(template.format(epoch, str(round(time.time() - vstime, 2)),
                              str(np.round(train_loss.result().numpy(), 4)),
                              str(np.round(train_auc.result().numpy(), 4)),
                              str(np.round(valid_loss.result().numpy(), 4)),
                              str(np.round(valid_auc.result().numpy(), 4)),
                              )
              )

    if c.cp_path:
        model.load_weights(c.cp_path)

    pred = []
    for tdata in test_dataset:
        p = model((tdata[0], tdata[1]), training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0)
    print(pred)
    print('Test AUC: ', roc_auc_score(y_true=y_test, y_score=pred))
    print('Test AP: ', average_precision_score(y_true=y_test, y_score=pred))



def train_cross_val(config):
    c = config
    r = c.data_name
    fold_num = 5
    cv_dir = c.cv_dir
    dire = "../../data/"
    total_loss = []
    total_auc = []
    total_Precision = []
    total_Recall = []
    total_ap = []
    iter = 0
    EPOCHS = c.epoch
    current_monitor = np.inf
    patient_count = 0


    idx_test = np.load(cv_dir + 'fold_split_idx.npy', allow_pickle=True)
    data_pos = np.load(dire + r + '_pos.npy',
                       allow_pickle=True).astype(np.float32)
    data_neg = np.load(dire + r + '_neg.npy',
                       allow_pickle=True).astype(np.float32)

    for iter in range(fold_num):

        @tf.function()
        def train_step(train_seq, train_nano, train_out):
            with tf.GradientTape() as tape:
                prob = model((train_seq, train_nano), training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                adam.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_auc(y_true=train_out, y_pred=prob)

        @tf.function()
        def valid_step(valid_seq, valid_nano, valid_out):
            prob = model((valid_seq, valid_nano), training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

            test_loss(vloss)
            test_auc(y_true=valid_out, y_pred=prob)
            test_precision(y_true=valid_out, y_pred=prob)
            test_recall(y_true=valid_out, y_pred=prob)

        print('Creating DeepOMe model for fold '+str(iter+1))
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

        adam = tfko.Adam(lr=c.lr_init, epsilon=1e-08, decay=c.lr_decay)
        train_loss = tfkm.Mean()
        test_loss = tfkm.Mean()
        train_auc = tfkm.AUC()
        test_auc = tfkm.AUC()
        test_precision = tfkm.Precision()
        test_recall = tfkm.Recall()

        print('Loading cv Data fold' + str(iter + 1))
        y_test = np.load(cv_dir + 'fold' + str(iter + 1) + '_label.npy',
                         allow_pickle=True).reshape(-1).astype(np.int32)
        test_nano = np.load(cv_dir + 'fold' + str(iter + 1) + '_nano.npy',
                            allow_pickle=True).astype(np.float32)
        test_seq = np.load(cv_dir + 'fold' + str(iter + 1) + '_seq.npy',
                           allow_pickle=True).astype(np.float32)

        train_pos = np.delete(data_pos, idx_test.item()['pos' + str(iter + 1)], axis=0)
        train_neg = np.delete(data_neg, idx_test.item()['neg' + str(iter + 1)], axis=0)
        train = np.concatenate([np.repeat(train_pos, 10, axis=0), train_neg], axis=0)
        train_nano = train[:, :164].reshape(-1, 4, 41).transpose([0, 2, 1]).astype(np.float32)
        nano_min = np.min(train_nano, axis=(0, 1))
        print(nano_min)
        nano_max = np.max(train_nano, axis=(0, 1))
        print(nano_max)
        train_nano = (train_nano - nano_min) / (nano_max - nano_min)
        
        train_seq = train[:, 164:].reshape(-1, 1001, 4).astype(np.float32)
        y_train = np.concatenate([np.ones(train_pos.shape[0]*10), np.zeros(train_neg.shape[0])]).astype(np.int32)
        train_seq = train_seq[:, 355:646, :]
        test_seq = test_seq[:, 355:646, :]
        sidx = np.random.permutation(y_train.shape[0])
        train_seq = train_seq[sidx]
        y_train = y_train[sidx]
        # train_nano[train_nano == -1] = 0
        # test_nano[test_nano == -1] = 0
        train_nano = train_nano[sidx]

        print(train_seq.shape)
        print(train_nano.shape)
        print(y_train.shape)
        print(np.mean(y_train))
        print(test_seq.shape)
        print(test_nano.shape)
        print(y_test.shape)
        print(np.mean(y_test))

        train_dataset = tfdd.from_tensor_slices((train_seq, train_nano, y_train))
        train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tfdd.from_tensor_slices((test_seq, test_nano, y_test))
        test_dataset = test_dataset.batch(128)
        print('Finish loading data')
        
        current_monitor = np.inf
        for epoch in tf.range(1, EPOCHS + 1):
            train_loss.reset_states()
            test_loss.reset_states()
            train_auc.reset_states()
            test_auc.reset_states()
            test_precision.reset_states()
            test_recall.reset_states()

            estime = time.time()
            for tdata in train_dataset:
                train_step(tdata[0], tdata[1], tdata[2])
            print(f'Training of epoch {epoch} finished! Time cost is {round(time.time() - estime, 2)}s')

            vstime = time.time()
            for vdata in test_dataset:
                valid_step(vdata[0], vdata[1], vdata[2])

            new_valid_monitor = np.round(test_loss.result().numpy(), 4)
            if new_valid_monitor < current_monitor:
                if c.cp_path:
                    model.save_weights(c.cp_path)
                    print('val_loss improved from {} to {}, saving model to {}'.
                          format(str(current_monitor), str(new_valid_monitor), c.cp_path))
                else:
                    print('val_loss improved from {} to {}, saving closed'.
                          format(str(current_monitor), str(new_valid_monitor)))

                current_monitor = new_valid_monitor
                patient_count = 0
            else:
                print('val_loss did not improved from {}'.format(str(current_monitor)))
                patient_count += 1

            if patient_count == 10:
                break

            template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}, VPr:{}, VRe:{}"
            print(template.format(epoch, str(round(time.time() - vstime, 2)),
                                  str(np.round(train_loss.result().numpy(), 4)),
                                  str(np.round(train_auc.result().numpy(), 4)),
                                  str(np.round(test_loss.result().numpy(), 4)),
                                  str(np.round(test_auc.result().numpy(), 4)),
                                  str(np.round(test_precision.result().numpy(), 4)),
                                  str(np.round(test_recall.result().numpy(), 4))
                                  )
                  )

        if c.cp_path:
            model.load_weights(c.cp_path)

        pred = []
        for tdata in test_dataset:
            p = model((tdata[0], tdata[1]), training=False)
            pred.append(p)
        pred = np.concatenate(pred, axis=0)
        test_auc = tfkm.AUC()
        test_auc(y_test, pred)
        test_precision = tfkm.Precision()
        test_recall = tfkm.Recall()
        test_precision(y_true=y_test, y_pred=pred)
        test_recall(y_true=y_test, y_pred=pred)
        test_ap = average_precision_score(y_true=y_test, y_score=pred)
        print('Test AUC: ', test_auc.result().numpy())
        print('Test Precision: ', test_precision.result().numpy())
        print('Test Recall: ', test_recall.result().numpy())
        print('Test AP: ', test_ap)

        total_loss.append(np.round(test_loss.result().numpy(), 4))
        total_auc.append(np.round(test_auc.result().numpy(), 4))
        total_Precision.append(np.round(test_precision.result().numpy(), 4))
        total_Recall.append(np.round(test_recall.result().numpy(), 4))
        total_ap.append(np.round(test_ap, 4))
        print(str(c.data_name)+' ' + str(c.model_funname) + '-------Fold '+str(iter+1)+' finished')
        iter = iter + 1

        pred = []
        for tdata in test_dataset:
            p = model((tdata[0], tdata[1]), training=False)
            pred.append(p)
        pred = np.concatenate(pred, axis=0)
        int_pred = (pred+0.5).astype(int)
        pidx = int_pred == 1
        nidx = int_pred == 0
        print('pred result has negative?:' + str(True in nidx))
        print('pred result hs positive?:' + str(True in pidx))

    print('loss')
    print(total_loss)
    print('auc')
    print(total_auc)
    print('precision')
    print(total_Precision)
    print('recall')
    print(total_Recall)
    print('ap')
    print(total_ap)
    
    template = "mean_loss: {}, mean_auc: {}, mean_precision:{}, mean_recall: {}, mean_ap:{}"
    print(template.format(str(np.mean(total_loss)),
                          str(np.mean(total_auc)),
                          str(np.mean(total_Precision)),
                          str(np.mean(total_Recall)),
                          str(np.mean(total_ap)),
                          )
          )
