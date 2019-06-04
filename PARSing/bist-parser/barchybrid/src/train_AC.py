# -*- coding:utf-8 -*-
'''

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

'''
from __future__ import print_function
import numpy as np
import tensorflow as tf
from critic_fold import Critic
from SDG_log import SDG
# np.random.seed(3435)  # for reproducibility, should be first
import os
import time
from keras.utils import np_utils
from attention import SimpleAttention, ContextAttention
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Merge, Dropout, RepeatVector, Permute

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bag_size = 1000
classes = 2
max_sen_len = 936
folds = 10
epochs = 1000
model_path = "../model/"
best_path = "../best_model/"
# best_path_critic = "./best_model/critic.best"
# best_path_actor = "./best_model/actor.best"
# batch_size = 50
nb_filter = 200
filter_length = 4
hidden_dims = nb_filter * 2
nb_epoch = 20
RNN = GRU
rnn_output_size = 100

lstm_output_size = 100

# parameters for critic:
filter_sizes = [3, 4, 5]
num_filters = 256
batch_size_critic = 5
GAMMA = 0.9
pre_epochs_critic = 1 

#####

# parameters for SDG:
# input size is determined after pre-training critic
hidden_size = 16
lr_SDG = 0.01
pre_epochs_actor = 1000 
batch_size_actor = 1
###

print('Loading data...')
import os
import sys
import datetime
import data_helper
import csv

file_dis_pre = open("prob_pre.csv", "w+")
file_dis = open("prob.csv", "w+")
out_path = "../out_local/"
out_path_folder2 = out_path + "all_sourcedata_cnn2/"
out_path_folder_f1 = out_path + "all_sourcedata_cnn_f1"
out_path_folder_f2 = out_path + "all_sourcedata_cnn_f2"

out_path_folder = out_path + "all_sourcedata_cnn"
if not os.path.exists(out_path_folder):
    os.makedirs(out_path_folder)

if not os.path.exists(out_path_folder2):
    os.makedirs(out_path_folder2)

if not os.path.exists(out_path_folder_f1):
    os.makedirs(out_path_folder_f1)

if not os.path.exists(out_path_folder_f2):
    os.makedirs(out_path_folder_f2)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(best_path):
    os.makedirs(best_path)
best_path_critic = best_path + "critic.best"
best_path_actor = best_path + "actor.best"

time_stamp = datetime.datetime.now()
time_str = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
root_dir = '../target_deal'
path1 = os.listdir(root_dir)
np.random.seed(10)

all_train = False
try:
    choose_target_str = sys.argv[1]
    choose_target = [choose_target_str]
except IndexError:
    all_train = True
    choose_target = ['target_dvd', 'target_books', 'target_kitchen', 'target_electronics']

choose_target=['target_dvd']
#choose_target = ['target_dvd']
for item_target in choose_target:
    for item in path1:
        if item_target in item:
            path = item
    print('now the target is ', item_target)
    model_path = model_path + item_target + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    sentence_lenth = int(path.split('_')[0])
    path = os.path.join(root_dir, path)

    # X_train, y_train, X_test, y_test, W, W2 = data_helper.load_data(fold=0, pad_left=True, path=path,
    #                                                                 max_lenth=sentence_lenth)
    # W represents the embeddings of all the words in the dataset. The number of rows denotes the number of words,
    # the number of columns denotes the feature dimension of embeddings.
    #     maxlen = X_train.shape[1]

    print('Train...')
    accs = []
    first_run = True
    # 'cross validation'
    for i in range(1):
        # for i in xrange(folds):
        X_train, y_train, X_test, y_test, W, W2 = data_helper.load_data(fold=i, pad_left=True, path=path,
                                                                        max_lenth=sentence_lenth)

        num_words = len(W)
        embedding_dims = len(W[0])

        # Shuffle training, dev, test sets
        rand_idx_train = np.random.permutation(range(len(X_train)))
        X_train = X_train[rand_idx_train]
        y_train = y_train[rand_idx_train]

        rand_idx_test = np.random.permutation(range(len(X_test)))
        X_test = X_test[rand_idx_test]
        y_test = y_test[rand_idx_test]
        X_dev = X_test[:len(X_test) // 2]
        y_dev = y_test[:len(y_test) // 2]
        X_test = X_test[len(X_test) // 2 + 1:]
        y_test = y_test[len(y_test) // 2 + 1:]

        y_test = np_utils.to_categorical(y_test, classes)
        y_train = np_utils.to_categorical(y_train, classes)
        y_dev = np_utils.to_categorical(y_dev, classes)

        maxlen = X_train.shape[1]
        num_train = X_train.shape[0]

        # for item in y_dev:
        #   print(item)

        sess = tf.Session()

        # pretrain the critic (target model)
        print("pretrain the critic...")
        critic = Critic(max_features=num_words, max_len=maxlen, embedding_dims=embedding_dims,
                        filter_sizes=filter_sizes, num_filters=num_filters, W=W, target_string=item_target,
                        time_str=time_str, GAMMA=GAMMA)
        critic.pre_train(X_dev, y_dev, X_test, y_test, 2, batch_size_critic, model_path=model_path)
        loss4test, acc4test = critic.test(X_test, y_test, batch_size_critic)

        print("pre_train loss4test:", loss4test, "pre_train acc4test:", acc4test)

        # pretrain the SDG (actor):
        print("pretrain the SDG...")
        file_dis_pre_writer = csv.writer(file_dis_pre)
        W_data = critic.get_repr(X_train)  # get the whole data representation
        data_features = W_data.shape[1]
        actor = SDG(sess, n_steps=bag_size, input_size=data_features, output_size=1, cell_size=hidden_size,
                    batch_size=batch_size_actor, lr=lr_SDG, repr=W_data)
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        var_ = {}
        #        for var in tf.all_variables():
        #            if "word_embedding" in var.name: continue
        #            if not var.name.startswith("Model"): continue
        #            var_[var.name.split(":")[0]] = var
        #        saver = tf.train.Saver(var_)

        createVar = globals()
        for bag_id in range(num_train // bag_size):
            createVar['g_' + str(bag_id)] =tf.Graph()
            with globals()['g_' + str(bag_id)].as_default():
                sess = tf.Session(graph=globals()['g_' + str(bag_id)])
                createVar['actor__' + str(bag_id)] =SDG(sess, n_steps=bag_size, input_size=data_features, output_size=1, cell_size=hidden_size,batch_size=batch_size_actor, lr=lr_SDG, repr=W_data)
                initializer = tf.global_variables_initializer()
                sess.run(initializer)
        best_acc = 0.0

        actor_list=[]
        epoch_acc=[]
        for actor_epoch in range(pre_epochs_actor):
            for bag_id in range(num_train // bag_size):
                #createVar['g_' + str(bag_id)] =tf.Graph()
                bag_start= bag_id * bag_size
                bag_end = min((bag_id + 1) * bag_size, num_train)
                cur_bag_size = bag_end - bag_start
                #state = np.ones(cur_bag_size)
                state=np.random.rand(cur_bag_size,2)
                X_train_t = X_train[bag_start:bag_end, :]
                y_train_t = y_train[bag_start:bag_end, :]
                #with globals()['g_' + str(bag_id)].as_default():
                    #sess = tf.Session(graph=globals()['g_' + str(bag_id)])
                    #createVar['actor__' + str(bag_id)] =SDG(sess, n_steps=bag_size, input_size=data_features, output_size=1, cell_size=hidden_size,batch_size=batch_size_actor, lr=lr_SDG, repr=W_data)
                    #initializer = tf.global_variables_initializer()
                    #sess.run(initializer)

#               for actor_epoch in range(pre_epochs_actor):
                print(state.shape,state)
                file_dis_pre_writer.writerow(state)
                W_data_t = critic.get_repr(X_train_t)
                    # print("---------------")
                    # print(np.reshape(W_data_t, [-1, bag_size, W_data_t.shape[1]]).shape)
                W_data_t_3d = np.reshape(W_data_t, [-1, bag_size, W_data_t.shape[1]])

                select_train, select_label = globals()['actor__' + str(bag_id)].sample(tf.convert_to_tensor(state), X_train_t, y_train_t)
                state_ = globals()['actor__' + str(bag_id)].deform(W_data_t_3d)
                state_=np.squeeze(state_)
                print("squ", state_)
                select_train_, select_label_ = globals()['actor__' + str(bag_id)].sample(tf.convert_to_tensor(state_), X_train_t, y_train_t)
                # print(select_train_)
                # print(select_label_)
                # print("*****state:", state)
                td_loss, acc, acc_ = critic.predict_self(select_train, select_label, select_train_, select_label_,X_dev,y_dev,epochs=pre_epochs_critic, batch_size=1)
                print("td_loss",td_loss)
                    #td_loss=0.05
                with globals()['g_' + str(bag_id)].as_default():
                     globals()['actor__' + str(bag_id)].learn(W_data_t_3d, td_loss)
                state = state_

                loss4test, acct = critic.test(X_test, y_test, batch_size_critic)
                if best_acc < acct:
                   best_acc = acct
                    # saver.save(sess, best_path_actor)
                print("bag_id", bag_id, "finish actor pretrain_epoch {} ...".format(actor_epoch), "acc_now", acct, "best_acc",best_acc)
            critic.model_update()
            print("updating critic...")
                
            loss4test, acct = critic.test(X_test, y_test, batch_size_critic)
            if best_acc < acct:
               best_acc = acct
            epoch_acc.append(acct)
            epoch_acc_t = np.array(epoch_acc)
            np.save(out_path+"epoch_acc.npy",epoch_acc_t)

        for ep in range(epochs):
            #state = np.ones(bag_size)
            #critic.load_W("../model/target_dvd/target_dvd2018.02.11-01:10:23--00-0.6601-0.6476.h5")
            for bag_id in range(num_train // bag_size):  # partition the dataset uniformly into bags of size: bag_size
                # to do: cluster the data representations to get bags.
                bag_start = bag_id * bag_size
                bag_end = min((bag_id + 1) * bag_size, num_train)
                cur_bag_size = bag_end - bag_start
                X_train_t = X_train[bag_start:bag_end, :]
                y_train_t = y_train[bag_start:bag_end, :]

                print("X_train_t.shape", X_train_t.shape)
                # print(len(X_train_t), 'train sequences')
                # print(len(X_test), 'test sequences')
                # print('X_train shape:', X_train_t.shape)
                W_data_t = critic.get_repr(X_train_t)
                W_data_t_3d = np.reshape(W_data_t, [-1, bag_size, W_data_t.shape[1]])
                select_train, select_label = globals()['actor__' + str(bag_id)].sample(tf.convert_to_tensor(state), X_train_t, y_train_t)
                # print(np.reshape(W_data, [-1, bag_size, W_data.shape[1]]).size)
                state_ = globals()['actor__' + str(bag_id)].deform(W_data_t_3d)
                select_train_, select_label_ = globals()['actor__' + str(bag_id)].sample(tf.convert_to_tensor(state_), X_train_t, y_train_t)
                td_error, acc, acc_= critic.predict_self(select_train, select_label, select_train_, select_label_, X_dev, y_dev,
                                        epochs=1,
                                        batch_size=1)
                globals()['actor__' + str(bag_id)].learn(W_data_t_3d, td_error)
                state = state_
                #if best_acc < acc or best_acc < acc_:
                    #best_acc = max(acc, acc_)
                loss4test, acc4test = critic.test(X_test, y_test, batch_size_critic)
                
                if best_acc < acc4test:
                    best_acc = acc4test
                print("epoch:", ep, "bag_id:", bag_id, "td_error:", td_error, "acc_now", acc, "acc_@now", acc_,
                      "best_acc", best_acc, "acc4test:", acc4test, "loss4test",loss4test)
                # saver.save(sess, best_path_actor)


                # print('X_test shape:', X_test.shape)
                # we need a good teacher, so the teacher should learn faster than the actor

                # sess.run(tf.global_variables_initializer())

                # if OUTPUT_GRAPH:
                #     tf.summary.FileWriter("logs/", sess.graph)

                # for i_episode in range(epochs):
                # s = env.reset()
                # t = 0
                # track_r = []
                # while True:
                #     if RENDER: env.render()
                #
                #     a = actor.choose_action(s)
                #
                #     s_, r, done, info = env.step(a)
                #
                #     if done: r = -20
                #
                #     track_r.append(r)
                #
                #     td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                #     actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
                #
                #     s = s_
                #     t += 1
                #
                #     if done or t >= epochs:
                #         ep_rs_sum = sum(track_r)
                #
                #         if 'running_reward' not in globals():
                #             running_reward = ep_rs_sum
                #         else:
                #             running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                #         if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                #         print("episode:", i_episode, "  reward:", int(running_reward))
                #         break

            # sequence_input = Input(shape=(maxlen,), dtype='int32')
            # model = Sequential()
            # embedding = Embedding(max_features, embedding_dims, weights=[np.matrix(W)], input_length=maxlen,
            #                       name='embedding')
            # embedded_sequences = embedding(sequence_input)
            # embedded_sequences = Dropout(0.4)(embedded_sequences)
            #
            # filter_sizes = [3, 4, 5]
            # num_filters = 256
            # conv_0 = Conv1D(num_filters, filter_sizes[0], activation='relu')(embedded_sequences)
            # conv_1 = Conv1D(num_filters, filter_sizes[1], activation='relu')(embedded_sequences)
            # conv_2 = Conv1D(num_filters, filter_sizes[2], activation='relu')(embedded_sequences)
            # maxpool_0 = MaxPooling1D(2)(conv_0)
            # maxpool_1 = MaxPooling1D(2)(conv_1)
            # maxpool_2 = MaxPooling1D(2)(conv_2)
            # merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
            # x = Conv1D(128, 3, activation='relu')(merged_tensor)
            # x = MaxPooling1D(2)(x)
            # x = Conv1D(128, 3, activation='relu')(x)
            # x = MaxPooling1D(2)(x)
            # x = GlobalMaxPooling1D()(x)
            # preds = Dense(2, activation='softmax')(x)
            # model = Model(sequence_input, preds)
            # model.summary()
            # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
            # get_nonlayer_model = Model(inputs=model.input, outputs=model.get_layer('global_max_pooling1d_1').output)
            #
            # get_nonlayer_model.summary()
            #
            # checkpointer = ModelCheckpoint(
            #     filepath="./out_local/all_sourcedata_cnn/" + '' + item_target + '_mod_cnn_time' + time_str + '-' + "-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5",
            #     save_weights_only=True, verbose=1)
            # model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test),
            #           callbacks=[checkpointer])
            # # 以下两行用于输入对于每个句子的向量表示
            # result_nonlayer = get_non_layer(x=X_test, model_nonlayer=get_nonlayer_model)
            # print(result_nonlayer)

            '''
            best_val_acc = 0
            best_test_acc = 0
            for j in xrange(nb_epoch):
                a = time.time()
                his = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            validation_split=0.1,
                            shuffle=True,
                            epochs=1, verbose=1)
                print('Fold %d/%d Epoch %d/%d\t%s' % (i + 1,
                                              folds, j + 1, nb_epoch, str(his.history)))
                if his.history['val_acc'][0] >= best_val_acc:
                    score, acc = model.evaluate(X_test, y_test,batch_size=batch_size,verbose=2)
                    best_val_acc = his.history['val_acc'][0]
                    best_test_acc = acc
                    print('Got best epoch  best val acc is %f test acc is %f' %(best_val_acc, best_test_acc))
                    if len(accs) > 0:
                        print('Current avg test acc:', str(np.mean(accs)))
                b = time.time()
                cost = b - a
                left = (nb_epoch - j - 1) + nb_epoch * (folds - i - 1)
            
            
                with open(log_path,'a') as f:
                    f.write('Current avg test acc:'+str(np.mean(accs))+'\n')
                print('One round cost %ds, %d round %ds %dmin left' % (cost, left,
                                                                   cost * left,
                                                                   cost * left / 60.0))
            accs.append(best_test_acc)
            print('Avg test acc:', str(np.mean(accs)))
    
            with open(log_path,'a') as f:
                f.write('Avg test acc:'+str(np.mean(accs))+'\n')
            '''
