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
                 W=None, target_string="test", time_str=None, batch_size=5, GAMMA=0.9, tau=0.001):
        """

        :rtype: object
        """

        self.tau = tau
        sequence_input = Input(shape=(max_len,), dtype='int32')
        self.model = Sequential()
        embedding = Embedding(max_features, embedding_dims, weights=[np.matrix(W)], input_length=max_len,
                              name='embedding')
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
        self.model.summary()
        self.model2 = self.model
        self.model_f1 = self.model
        self.model_f2 = self.model
        # self.v_ = tf.placeholder()  #State evaluation for the former state
        self.item_target = target_string
        self.time_str = time_str
        # self.batch_size = batch_size
        # self.epochs = epochs
        self.GAMMA = GAMMA
        
        self.weights_all_history = {}
        # self.v=0

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

        self.model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

        self.model_f1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

        self.model_f2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

    def learn(self, X_train, y_train, X_train_, y_train_, X_dev, y_dev, epochs, batch_size):
        #  for learning interacting with SDG(actor)
        print("Critic is learning with the actor...")

        #        self.model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
        checkpointer = ModelCheckpoint(
            filepath="../out_local/all_sourcedata_cnn/" + '' + self.item_target + '_mod_cnn_time' + self.time_str + '-' +
                     "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)

        checkpointer2 = ModelCheckpoint(
            filepath="../out_local/all_sourcedata_cnn2/" + '' + self.item_target + '_mod_cnn_time' + self.time_str + '-' +
                      "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                              validation_data=(X_dev, y_dev), callbacks=[checkpointer])

        hist_ = self.model2.fit(X_train_, y_train_, batch_size=batch_size, epochs=epochs,
                                validation_data=(X_dev, y_dev), callbacks=[checkpointer2])
        self.v = hist.history['val_loss']
        self.v_ = hist_.history['val_loss']
        # self.v_,_ = self.model.evaluate(X_train_, y_train_, batch_size=batch_size, verbose=2)
        # self.v_ = hist_.history[val_loss]
        self.td_error = self.GAMMA * self.v_ - self.v
        acc = hist.history['val_acc']
        acc_ = hist_.history['val_acc']
        return self.td_error, acc, acc_

    def test(self, X_test, y_test, batch_size):
        print("Critic is testing on the test data...")
        loss4test, acc1 = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
        return loss4test, acc1

    def pre_train(self, X_train, y_train, X_dev, y_dev, epochs, batch_size, model_path):  # for pre-training itself
        print("Critic is pre_training...")
        #        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
        checkpointer = ModelCheckpoint(
            filepath=model_path + '' + self.item_target + self.time_str + '-' +
                     "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_dev, y_dev), callbacks=[checkpointer])

        self.weights_all_history = {}
        self.model_layer = self.model.layers
        for layer in range(len(self.model_layer)):
            weight = self.model_layer[layer].get_weights()
            for item_list in range(len(weight)):
                name = str(layer) + '_' + str(item_list)
                self.weights_all_history[name] = weight[item_list]

    def train(self, X_train, y_train, X_dev, y_dev, epochs, batch_size, model_path):  # for pre-training itself
        print("Critic is training...")
        #        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
        checkpointer = ModelCheckpoint(
            filepath=model_path + '' + self.item_target + self.time_str + '-' +
                     "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_dev, y_dev), callbacks=[checkpointer])
        self.weight_list_new_now = {}
        
        self.model_layer = self.model.layers
        for layer in range(len(self.model_layer)):
            weight = self.model_layer[layer].get_weights()
            save_model_fin = []
            for item_list in range(len(weight)):
                name = str(layer) + '_' + str(item_list)
                weight_list_new = weight[item_list] * self.tau + self.weights_all_history[name]
                save_model_fin.append(weight_list_new)
                self.weight_list_new_now[name] = weight_list_new
            self.model_layer[layer].set_weights(save_model_fin)
        self.weights_all_history = self.weight_list_new_now

    def predict_self(self, X_test, y_test, X_test_, y_test_, X_dev, y_dev, epochs,
                batch_size):  # for freezing training and SDG pre-training
        #        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

        checkpointer_f1 = ModelCheckpoint(
            filepath="../out_local/all_sourcedata_cnn_f1/" + '' + self.item_target + self.time_str + '-' +
                     "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)

        checkpointer_f2 = ModelCheckpoint(
            filepath="../out_local/all_sourcedata_cnn_f2/" + '' + self.item_target + self.time_str + '-' +
                     "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5", save_weights_only=True, verbose=1)
        hist_f1 = self.model_f1.fit(X_test, y_test, batch_size=1, epochs=epochs,
                                    validation_data=(X_dev, y_dev), callbacks=[checkpointer_f1])

        hist_f2 = self.model_f2.fit(X_test_, y_test_, batch_size=1, epochs=epochs,
                                    validation_data=(X_dev, y_dev), callbacks=[checkpointer_f2])
        self.model_f1 = self.model
        self.model_f2 = self.model
        print("########hist_f2: ", hist_f2.history['val_loss'])

        print("########hist_f1: ", hist_f1.history['val_loss'])
        # test_loss, acc = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
        # test_loss_, acc_ = self.model.evaluate(X_test_, y_test_, batch_size=batch_size, verbose=2)
        self.td_loss = self.GAMMA * hist_f2.history['val_loss'][0] - hist_f1.history['val_loss'][0]
        acc = hist_f1.history['val_acc']
        return self.td_loss, acc

    def predict_get_prob(self, x_in):
        prob = self.model.predict(x_in)
        return prob

    def get_layer_result(self, x=None, model_layer=None):
        layer_result = model_layer.predict(x)
        return layer_result

    def get_repr(self, x_in):
        get_nonlayer_model = Model(inputs=self.model.input,
                                   outputs=self.model.get_layer('global_max_pooling1d_1').output)
        # get_nonlayer_model.summary()
        result_nonlayer = self.get_layer_result(x=x_in, model_layer=get_nonlayer_model)
        return result_nonlayer

    def save(self, best_path):
        self.model.save_weights(best_path + "_weights.h5")
        json_string = self.model.to_json()
        open(best_path, 'w').write(json_string)

    def keras_soft_replace(self):
        tau = 0.1
