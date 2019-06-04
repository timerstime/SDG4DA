from __future__ import print_function
import numpy as np
# np.random.seed(3435)  # for reproducibility, should be first
import os

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, \
    Embedding, Convolution1D, MaxPooling1D, AveragePooling1D, \
    Input, Dense, merge
from keras.regularizers import l2
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.constraints import maxnorm
from keras.datasets import imdb
from keras import callbacks
from keras.utils import generic_utils
from keras.models import Model
from keras.optimizers import Adadelta
import time
from keras.utils import np_utils
from attention import SimpleAttention, ContextAttention
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Merge, Dropout, RepeatVector, Permute
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Dense, Activation, Input, merge


class Critic(object):
    def __init__(self, max_features=300, max_len=936, embedding_dims=100, filter_sizes=None, num_filters=256,
                 W=None, target_dataset="test", time_str=None, batch_size=10, epochs=1):
        sequence_input = Input(shape=(max_len,), dtype='int32')
        self.model = Sequential()
        embedding = Embedding(max_features, embedding_dims, weights=[np.matrix(W)], input_length=max_len, name='embedding')
        embedded_sequences = embedding(sequence_input)
        embedded_sequences = Dropout(0.4)(embedded_sequences)
        conv_0 = Conv1D(num_filters, filter_sizes[0], activation='relu')(embedded_sequences)
        conv_1 = Conv1D(num_filters, filter_sizes[1], activation='relu')(embedded_sequences)
        conv_2 = Conv1D(num_filters, filter_sizes[2], activation='relu')(embedded_sequences)
        maxpool_0 = MaxPooling1D(2)(conv_0)
        maxpool_1 = MaxPooling1D(2)(conv_1)
        maxpool_2 = MaxPooling1D(2)(conv_2)
        merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
        x = Conv1D(128, 3, activation='relu')(merged_tensor)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = GlobalMaxPooling1D()(x)
        preds = Dense(2, activation='softmax')(x)
        self.model = Model(sequence_input, preds)
        self._v = 0   #State evaluation for the former state
        self.item_target = target_dataset
        self.time_str = time_str

    def train(self, X_train, y_train, X_train_, y_train_, X_dev, y_dev):
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
        checkpointer = ModelCheckpoint(
        filepath="./out_local/all_sourcedata_cnn/" + '' + self.item_target + '_mod_cnn_time' + self.time_str + '-' + "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5",
        save_weights_only=True, verbose=1)
        hist=self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=epochs, validation_data=(X_dev, y_dev),
              callbacks=[checkpointer])

    def get_non_layer(self, x=None, model_nonlayer=None):
        result_nonlayer = model_nonlayer.predict(x)
        return result_nonlayer

    def get_repr(self, x_in ):
        get_nonlayer_model = self.model(inputs=self.model.input, outputs=self.model.get_layer('global_max_pooling1d_1').output)
        get_nonlayer_model.summary()
        result_nonlayer = self.get_non_layer(x=x_in, model_nonlayer=get_nonlayer_model)
        return result_nonlayer
