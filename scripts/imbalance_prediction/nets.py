import tensorflow as tf
from tensorflow.keras.regularizers import l2

tfk = tf.keras
tfkl = tf.keras.layers


class ResBlock2(tf.keras.Model):
    def __init__(self):
        super(ResBlock2, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn1 = tfkl.BatchNormalization()
        self.avt = tfkl.ReLU()
        self.conv2 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn2 = tfkl.BatchNormalization()
        self.conv3 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn3 = tfkl.BatchNormalization()
        self.conv4 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn4 = tfkl.BatchNormalization()
        self.conv5 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn5 = tfkl.BatchNormalization()
        self.conv6 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.bn6 = tfkl.BatchNormalization()
        self.conv7 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')
        self.conv8 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2, activation='relu')

    def call(self, inputs, training=None, mask=None):
        h1 = self.conv1(inputs)
        h2 = self.bn1(h1, training=training)
        h2 = self.conv2(h2)
        h2 = self.bn2(h2)
        h2 = self.conv3(h2)
        h3 = tfkl.Add()([h1, h2])
        h4 = self.bn3(h3)
        h4 = self.conv4(h4)
        h4 = self.bn4(h4)
        h4 = self.conv5(h4)
        h5 = tfkl.Add()([h3, h4])
        h6 = self.bn5(h5)
        h6 = self.conv6(h6)
        h6 = self.bn6(h6)
        h6 = self.conv7(h6)
        h7 = tfkl.Add()([h5, h6])
        h7 = self.conv8(h7)
        return h7


class ResBlock(tf.keras.Model):
    def __init__(self):
        super(ResBlock, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn1 = tfkl.BatchNormalization()
        self.avt = tfkl.ReLU()
        self.conv2 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn2 = tfkl.BatchNormalization()
        self.conv3 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn3 = tfkl.BatchNormalization()
        self.conv4 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn4 = tfkl.BatchNormalization()
        self.conv5 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn5 = tfkl.BatchNormalization()
        self.conv6 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.bn6 = tfkl.BatchNormalization()
        self.conv7 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)
        self.conv8 = tfkl.Conv1D(32, 10, padding='same', dilation_rate=2)

    def call(self, inputs, training=None, mask=None):
        h1 = self.conv1(inputs)
        h2 = self.bn1(h1, training=training)
        h2 = self.avt(h2)
        h2 = self.conv2(h2)
        h2 = self.bn2(h2, training=training)
        h2 = self.avt(h2)
        h2 = self.conv3(h2)
        h3 = tfkl.Add()([h1, h2])
        h4 = self.bn3(h3, training=training)
        h4 = self.avt(h4)
        h4 = self.conv4(h4)
        h4 = self.bn4(h4, training=training)
        h4 = self.avt(h4)
        h4 = self.conv5(h4)
        h5 = tfkl.Add()([h3, h4])
        h6 = self.bn5(h5, training=training)
        h6 = self.avt(h6)
        h6 = self.conv6(h6)
        h6 = self.bn6(h6, training=training)
        h6 = self.avt(h6)
        h6 = self.conv7(h6)
        h7 = tfkl.Add()([h5, h6])
        h7 = self.conv8(h7)
        h7 = self.avt(h7)
        return h7


# Model only using sequence data
class create_model1(tf.keras.Model):
    def __init__(self):
        super(create_model1, self).__init__()

        self.left_conv1 = tfkl.Conv1D(64, 7, padding='valid', activation='relu')
        self.left_pool1 = tfkl.MaxPool1D(10, 4)
        self.left_drop1 = tfkl.Dropout(0.25)

        self.conv_merged = tfkl.Conv1D(8, 5, padding='valid', activation='relu')
        self.merged_pool = tfkl.MaxPool1D(10, 5)
        self.merged_drop = tfkl.Dropout(0.25)

        self.hidden1 = tfkl.Dense(5, activation='relu')
        self.out = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        h = self.left_conv1(x1)
        h = self.left_pool1(h)
        h = self.left_drop1(h, training=training)
        h = self.conv_merged(h)
        h = self.merged_pool(h)
        h = self.merged_drop(h, training=training)
        h = tfkl.Flatten()(h)
        h = self.hidden1(h)
        out = self.out(h)
        return out


# Model using both sequence and nanopore features
class create_model2(tf.keras.Model):
    def __init__(self):
        super(create_model2, self).__init__()

        self.left_conv1 = tfkl.Conv1D(64, 7, padding='valid', activation='relu')
        self.left_pool1 = tfkl.MaxPool1D(10, 4)
        self.left_drop1 = tfkl.Dropout(0.25)

        self.right_conv1 = tfkl.Conv1D(64, 7, padding='valid', activation="relu")
        self.right_pool1 = tfkl.MaxPool1D(10, 5)
        self.right_drop1 = tfkl.Dropout(0.25)

        self.conv_merged = tfkl.Conv1D(8, 5, padding='valid', activation='relu')
        self.merged_pool = tfkl.MaxPooling1D(10, 5)
        self.merged_drop = tfkl.Dropout(0.25)

        self.hidden1 = tfkl.Dense(5, activation='relu')
        self.out = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        h1 = self.left_conv1(x1)
        h1 = self.left_pool1(h1)
        h1 = self.left_drop1(h1, training=training)

        h2 = self.right_conv1(x2)
        h2 = self.right_pool1(h2)
        h2 = self.right_drop1(h2, training=training)

        h = tf.concat([h1, h2], axis=-2)
        h = self.conv_merged(h)
        h = self.merged_pool(h)
        h = self.merged_drop(h, training=training)
        h = tfkl.Flatten()(h)
        h = self.hidden1(h)
        out = self.out(h)
        return out


# Nanopore features directly sent to output layers
class create_model3(tf.keras.Model):
    def __init__(self):
        super(create_model3, self).__init__()

        self.left_conv1 = tfkl.Conv1D(64, 7, padding='valid', activation='relu')
        self.left_pool1 = tfkl.MaxPool1D(10, 4)
        self.left_drop1 = tfkl.Dropout(0.25)

        self.right_conv1 = tfkl.Conv1D(64, 7, padding='valid', activation="relu")
        self.right_pool1 = tfkl.MaxPool1D(10, 5)
        self.right_drop1 = tfkl.Dropout(0.25)

        self.conv_merged = tfkl.Conv1D(8, 5, padding='valid', activation='relu')
        self.merged_pool = tfkl.MaxPooling1D(10, 5)
        self.merged_drop = tfkl.Dropout(0.25)

        self.hidden1 = tfkl.Dense(5, activation='relu')
        self.out = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        h1 = self.left_conv1(x1)
        h1 = self.left_pool1(h1)
        h1 = self.left_drop1(h1, training=training)
        h1 = self.conv_merged(h1)
        h1 = self.merged_pool(h1)
        h1 = self.merged_drop(h1, training=training)
        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(x2)

        h = tf.concat([h1, h2], axis=-1)
        h = self.hidden1(h)
        out = self.out(h)
        return out


# Baseline model iRNA_PseKNC
class create_model4(tf.keras.Model):
    def __init__(self):
        super(create_model4, self).__init__()

        self.conv1 = tfkl.Conv1D(8, 4, strides=2, padding='valid', activation='relu')
        self.conv2 = tfkl.Conv1D(4, 2, strides=2, padding='valid', activation='relu')
        self.drop = tfkl.Dropout(0.35)

        self.hidden1 = tfkl.Dense(3, activation='relu')
        self.out = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        h = self.conv1(x1)
        h = self.conv2(h)
        h = self.drop(h, training=training)
        h = tfkl.Flatten()(h)
        h = self.hidden1(h)
        out = self.out(h)
        return out


class DeepOMe2(tf.keras.Model):
    def __init__(self):
        super(DeepOMe2, self).__init__()

        self.stem1 = tfkl.Conv1D(32, 1, padding='same', activation='relu')
        self.stem2 = tfkl.Conv1D(32, 3, padding='same', activation='relu')
        self.stem3 = tfkl.Conv1D(32, 5, padding='same', activation='relu')

        self.bn1 = tfkl.BatchNormalization()
        self.bn2 = tfkl.BatchNormalization()
        self.bn3 = tfkl.BatchNormalization()
        self.avt = tfkl.ReLU()

        self.res = ResBlock()
        self.bilstm1 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2, return_sequences=True))
        self.bilstm2 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2))
        self.out = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        x1 = x1[:, 355:646, :]
        h1 = self.stem1(x1)
        h1 = self.bn1(h1)
        h2 = self.stem2(x1)
        h2 = self.bn2(h2)
        h3 = self.stem3(x1)
        h3 = self.bn3(h3)
        h = tf.concat([h1, h2, h3], axis=-1)

        h = self.res(h)
        h = self.bilstm1(h)
        h = self.bilstm2(h)
        h = self.out(h)

        return h


class DeepOMe(tf.keras.Model):
    def __init__(self):
        super(DeepOMe, self).__init__()

        self.stem1 = tfkl.Conv1D(32, 1, padding='same')
        self.stem2 = tfkl.Conv1D(32, 3, padding='same')
        self.stem3 = tfkl.Conv1D(32, 5, padding='same')

        self.bn1 = tfkl.BatchNormalization()
        self.bn2 = tfkl.BatchNormalization()
        self.bn3 = tfkl.BatchNormalization()
        self.avt = tfkl.ReLU()

        self.res = ResBlock()
        self.bilstm1 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2, return_sequences=True))
        self.bilstm2 = tfkl.Bidirectional(tfkl.LSTM(32, dropout=0.2, return_sequences=True))
        self.out = tfkl.Conv1D(2, 1, activation='softmax')

    def call(self, inputs, training=True, mask=None):
        x1, x2 = inputs
        # x1 = x1[:, 355:646, :]
        h1 = self.stem1(x1)
        h1 = self.bn1(h1, training=training)
        h1 = self.avt(h1)
        h2 = self.stem2(x1)
        h2 = self.bn2(h2, training=training)
        h2 = self.avt(h2)
        h3 = self.stem3(x1)
        h3 = self.bn3(h3, training=training)
        h3 = self.avt(h3)
        h = tf.concat([h1, h2, h3], axis=-1)

        h = self.res(h)
        h = self.bilstm1(h)
        h = self.bilstm2(h)
        h = self.out(h)

        return h[:, 145, 1]


class TSDL(tf.keras.Model):
    def __init__(self):
        super(TSDL, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool = tfkl.MaxPool1D(10, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv4 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv6 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool(h1)
        h1 = self.dropout(h1)
        h2 = self.conv3(h1)
        h2 = self.conv4(h2)
        h2 = self.pool(h2)
        h2 = self.dropout(h2)
        h3 = self.conv5(h2)
        h3 = self.conv6(h3)
        h3 = self.pool(h3)
        h3 = self.dropout(h3)

        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(h2)
        h3 = tfkl.Flatten()(h3)
        h = tf.concat([h1, h2, h3], axis=1)
        h = self.fc1(h)
        out = self.fc2(h)
        return out


class Seq2pO1(tf.keras.Model):
    def __init__(self):
        super(Seq2pO1, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 5)
        self.pool2 = tfkl.MaxPool1D(10, 5)
        self.dropout = tfkl.Dropout(0.6)
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

        self.nano_conv = tfkl.Conv1D(32, 7, padding='same', activation="relu")
        self.nano_pool = tfkl.MaxPool1D(2, 2)
        self.nano_drop = tfkl.Dropout(0.25)

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool1(h1)
        h1 = self.dropout(h1)
        h2 = self.rnn1(h1)
        h2 = self.pool2(h2)
        h2 = self.dropout(h2)

        h2 = tfkl.Flatten()(h2)

        n = self.nano_conv(nano)
        n = self.nano_pool(n)
        n = self.nano_drop(n)
        n = tfkl.Flatten()(n)

        h = tf.concat([h2, n], axis=1)
        h = self.fc1(h)
        out = self.fc2(h)
        return out


class Seq2pO(tf.keras.Model):
    def __init__(self):
        super(Seq2pO, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 4)
        self.pool2 = tfkl.MaxPool1D(4, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv4 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        # self.conv6 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
        #                          kernel_regularizer=l2(0.001))

        # self.conv2 = tfkl.Bidirectional(tfkl.LSTM(
        #     16, dropout=0.6,kernel_regularizer=l2(0.001), return_sequences=True))
        # self.conv4 = tfkl.Bidirectional(tfkl.LSTM(
        #     16, dropout=0.6, kernel_regularizer=l2(0.001), return_sequences=True))
        # self.conv6 = tfkl.Bidirectional(tfkl.LSTM(
        #     16, dropout=0.6, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool1(h1)
        h1 = self.dropout(h1)
        h2 = self.conv3(h1)
        h2 = self.rnn2(h2)
        h2 = self.pool2(h2)
        # h2 = self.dropout(h2)
        h3 = self.conv5(h2)
        h3 = self.rnn3(h3)
        h3 = self.pool2(h3)
        # h3 = self.dropout(h3)

        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(h2)
        h3 = tfkl.Flatten()(h3)
        h = tf.concat([h1, h2, h3], axis=1)

        h = self.fc1(h)
        out = self.fc2(h)
        return out


# class Nano2pO(tf.keras.Model):
#     def __init__(self):
#         super(Nano2pO, self).__init__()
#
#         self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.pool1 = tfkl.MaxPool1D(10, 4)
#         self.pool2 = tfkl.MaxPool1D(4, 4)
#         self.dropout = tfkl.Dropout(0.6)
#         self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.conv4 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.conv6 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
#                                  kernel_regularizer=l2(0.001))
#         self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
#             16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
#         self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
#             16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
#         self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
#             16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
#
#         self.nano_conv = tfkl.Conv1D(32, 7, padding='same', activation="relu")
#         self.nano_pool = tfkl.MaxPool1D(2, 2)
#         self.nano_drop = tfkl.Dropout(0.25)
#
#         self.fc0 = tfkl.Dense(256, activation='relu')
#         self.fc1 = tfkl.Dense(128, activation='relu')
#         self.fc2 = tfkl.Dense(1, activation='sigmoid')
#
#     def call(self, inputs, training=True, mask=None):
#         seq, nano = inputs
#         h1 = self.conv1(seq)
#         h1 = self.conv2(h1)
#         h1 = self.pool1(h1)
#         h1 = self.dropout(h1)
#         h2 = self.conv3(h1)
#         h2 = self.conv4(h2)
#         h2 = self.pool2(h2)
#         h2 = self.dropout(h2)
#         h3 = self.conv5(h2)
#         h3 = self.conv6(h3)
#         h3 = self.pool2(h3)
#         h3 = self.dropout(h3)
#
#         h1 = tfkl.Flatten()(h1)
#         h2 = tfkl.Flatten()(h2)
#         h3 = tfkl.Flatten()(h3)
#
#         n = self.nano_conv(nano)
#         n = self.nano_pool(n)
#         n = self.nano_drop(n)
#         n = tfkl.Flatten()(n)
#
#         h = tf.concat([h1, h2, h3, n], axis=1)
#         h = self.fc1(h)
#         out = self.fc2(h)
#         return out


class NanoOnly(tf.keras.Model):
    def __init__(self):
        super(NanoOnly, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 4)
        self.pool2 = tfkl.MaxPool1D(4, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.nano_conv = tfkl.Conv1D(32, 7, padding='same', activation="relu")
        # self.nano_conv2 = tfkl.Conv1D(32, 3, padding='same', activation="relu")

        self.nano_pool = tfkl.MaxPool1D(2, 2)
        self.nano_drop = tfkl.Dropout(0.25)

        self.fc1 = tfkl.Dense(5, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs

        n = self.nano_conv(nano)
        n = self.nano_pool(n)
        # n = self.nano_drop(n)
        n = tfkl.Flatten()(n)

        h = n
        h = self.fc1(h)
        out = self.fc2(h)
        return out


class Nano2pO(tf.keras.Model):
    def __init__(self):
        super(Nano2pO, self).__init__()

        self.conv1 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.pool1 = tfkl.MaxPool1D(10, 4)
        self.pool2 = tfkl.MaxPool1D(4, 4)
        self.dropout = tfkl.Dropout(0.6)
        self.conv3 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.conv5 = tfkl.Conv1D(64, 7, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.001))
        self.rnn1 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn2 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))
        self.rnn3 = tfkl.Bidirectional(tfkl.LSTM(
            16, dropout=0.2, kernel_regularizer=l2(0.001), return_sequences=True))

        self.nano_conv = tfkl.Conv1D(32, 7, padding='same', activation="relu")
        # self.nano_conv2 = tfkl.Conv1D(32, 3, padding='same', activation="relu")

        self.nano_pool = tfkl.MaxPool1D(2, 2)
        self.nano_drop = tfkl.Dropout(0.25)

        self.fc1 = tfkl.Dense(256, activation='relu')
        self.fc2 = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        seq, nano = inputs
        h1 = self.conv1(seq)
        h1 = self.conv2(h1)
        h1 = self.pool1(h1)
        h1 = self.dropout(h1)
        h2 = self.conv3(h1)
        h2 = self.rnn2(h2)
        h2 = self.pool2(h2)
        h2 = self.dropout(h2)
        h3 = self.conv5(h2)
        h3 = self.rnn3(h3)
        h3 = self.pool2(h3)
        h3 = self.dropout(h3)

        h1 = tfkl.Flatten()(h1)
        h2 = tfkl.Flatten()(h2)
        h3 = tfkl.Flatten()(h3)

        n = self.nano_conv(nano)
        n = self.nano_pool(n)
        # n = self.nano_drop(n)
        n = tfkl.Flatten()(n)

        h = tf.concat([h1, h2, h3, n], axis=1)
        # h = n
        h = self.fc1(h)
        out = self.fc2(h)
        return out
