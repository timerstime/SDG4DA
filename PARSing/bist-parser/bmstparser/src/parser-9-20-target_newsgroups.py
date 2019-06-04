# -*- coding: utf-8 -*-
from optparse import OptionParser
import pickle, utils, mstlstm, os, os.path, time

#本任务分类数目:12
import pickle
import argparse
import random
import time
import sys
import numpy as np
import os
import pickle
import dynet
import codecs
random.seed(10)
np.random.seed(10)


import numpy as np
import tensorflow as tf
from SDG_log import SDG
np.random.seed(3435)  # for reproducibility, should be first
import os
import time
#from keras.utils import np_utils
#from attention import SimpleAttention, ContextAttention
#from keras.layers import Embedding, Bidirectional, LSTM, GRU, Merge, Dropout, RepeatVector, Permute

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
bag_size = 1000
#classes = 2
#max_sen_len = 936
#folds = 10
epochs = 1000
model_path = "../model/"
best_path = "../best_model/"

batch_size_critic = 5
GAMMA = 0.9
pre_epochs_critic = 1

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
#import data_helper
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
root_dir = '../save_parsing'
path1 = os.listdir(root_dir)


all_train = False
try:
    choose_target_str = sys.argv[1]
    choose_target = [choose_target_str]
except IndexError:
    all_train = True
    choose_target = ['target_weblogs','target_wsj',  'target_reviews', 'target_answers','target_newsgroups','target_emails']

choose_target=['target_newsgroups']

for item_target in choose_target:
    for item in path1:
        if item_target in item:
            path = item
    print('now the target is ', item_target)
    model_path = model_path + item_target + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    path = os.path.join(root_dir, path)

    path_source = os.path.join(path, 'train_dev.conll')
    path_dev = os.path.join(path, 'test2dev.conll')
    path_test = os.path.join(path, 'test2test.conll')
    print('path_source:', path_source)
    print('path_dev:', path_dev)
    print('path_test:', path_test)

    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=path_source)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default=path_dev)
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default=path_test)

    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE",default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="./out")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()
    #使用了外部embedding层
    print 'Using external embedding:', options.external_embedding

    if not options.predictFlag:

        print 'Preparing vocab'
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, options)
        pre_train_epoch=2
        train_data = parser.get_data_as_indices(path_source)
        #W_data = parser.get_repr_self(train_X)

        best_acc=0.0
        for epoch in xrange(pre_train_epoch):
            print 'Starting epoch', epoch
            parser.Train_self(train_data)
            conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
            testpath = os.path.join(options.output,'test_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(testpath, parser.Predict(path_test))
            parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch+1)))
            print("predicting....")
            if not conllu:
                os.system('perl ./utils/eval.pl -g ' + path_test + ' -s ' + testpath+ ' > ' +'temp' + '.txt')
            else:
                os.system(
                    'python ./utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + path_test + ' ' + testpath+ ' > ' +'temp' + '.txt')

            with open('./temp.txt','r') as f:
                iii=0
                for line in f.readlines():
                    iii+=1
                    if iii>4:
                        break
                    else:
                        if iii==1:
                            line_las=line.replace('\n','')
                            line_las=line_las.replace('%','')
                            line_las=line_las.split('=')
                            line_las=float(line_las[1])
                            acc=line_las
                            if best_acc<acc:
                               best_acc=acc
                        print(line.replace('\n',''))
            print("best_las",best_acc) 

        print("pretrain the SDG...")
        file_dis_pre_writer = csv.writer(file_dis_pre)
        sess = tf.Session()

        #initializer = tf.global_variables_initializer()
        #sess.run(initializer)

        

        print("load all data and shuffle the source data...")
        train_X= parser.get_data_as_indices(path_source)           #!!!!!!Warning: should it be changed into "path_train"?

        rand_idx_train = np.random.permutation(range(len(train_X)))
        train_X_temp = np.array(train_X)[rand_idx_train]
        train_X = train_X_temp
        # 去掉不能被1000整除的部分，以保证不会出错
        print("should cut some train_data", len(train_X) % bag_size)
        should_cut = len(train_X) - len(train_X) % bag_size
        train_X = train_X[:should_cut]
        dev_X = parser.get_data_as_indices(path_dev)
        test_X= parser.get_data_as_indices(path_test)

        W_data = parser.get_repr_self(train_X) ##!!!Warning: used to be test_X
        data_features = W_data.shape[1]


        actor = SDG(sess, n_steps=bag_size, input_size=data_features, output_size=1, cell_size=hidden_size,
                    batch_size=batch_size_actor, lr=lr_SDG, repr=W_data)
        # 以下变量为保持名称一致
        X_train = train_X

        X_dev = dev_X

        num_train = len(X_train)
        '''
        createVar = globals()
        for bag_id in range(num_train // bag_size):
            createVar['g_' + str(bag_id)] = tf.Graph()
            with globals()['g_' + str(bag_id)].as_default():
                sess = tf.Session(graph=globals()['g_' + str(bag_id)])
                createVar['actor__' + str(bag_id)] = SDG(sess, n_steps=bag_size, input_size=data_features,
                                                         output_size=1,
                                                         cell_size=hidden_size, batch_size=batch_size_actor, lr=lr_SDG,
                                                         repr=W_data)
        '''
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        var_ = {}
        best_acc = 0.0
        actor_list = []
        epoch_acc = []


        for actor_epoch in range(pre_epochs_actor):
            save_acc_loss_dict={}
            save_acc_loss_dict_path=open(os.path.join(save4dir,str(actor_epoch)),'wb')
            acc_save={}
            tderror_save={}
            select_id={}
            best_save={}
            W_data_t_save={}
            state_t_save={}
            X_data_t_dev_save={}
            select_count_save={}
            loss4test_save={}
            select_num1_save={}
            select_num2_save={}
            td_sum_save={}
            td_sum=0
            for bag_id in range(num_train // bag_size):
                # createVar['g_' + str(bag_id)] =tf.Graph()
                bag_start = bag_id * bag_size
                bag_end = min((bag_id + 1) * bag_size, num_train)
                cur_bag_size = bag_end - bag_start
                # state = np.ones(cur_bag_size)
                state = np.random.rand(cur_bag_size, 2)
                X_train_t = X_train[bag_start:bag_end, :]  ##!!!warning: former version has no y_train_t, indice [bag_start:bag_end]
                y_train_t = y_train[bag_start:bag_end, :]
                # with globals()['g_' + str(bag_id)].as_default():
                # sess = tf.Session(graph=globals()['g_' + str(bag_id)])
                # createVar['actor__' + str(bag_id)] =SDG(sess, n_steps=bag_size, input_size=data_features, output_size=1, cell_size=hidden_size,batch_size=batch_size_actor, lr=lr_SDG, repr=W_data)
                # initializer = tf.global_variables_initializer()
                # sess.run(initializer)

                #               for actor_epoch in range(pre_epochs_actor):
                print(state.shape, state)
                file_dis_pre_writer.writerow(state)
                W_data_t = parser.get_repr_self(X_train_t)

                if bag_id == 0:
                    select_train = X_train_t
                    select_label = y_train_t

                X_data_t_dev=critic.get_repr(X_dev)
                W_data_t_save["W_data_t"+"_bag_id_"+str(bag_id)]=W_data_t
                X_data_t_dev_save["X_data_t_dev"+"_bag_id_"+str(bag_id)]=X_data_t_dev

                W_data_t_3d = np.reshape(W_data_t, [-1, bag_size, W_data_t.shape[1]])

                #select_train, select_label = globals()['actor__' + str(bag_id)].sample(tf.convert_to_tensor(state),X_train_t)
                state_ = actor.deform(W_data_t_3d)
                state_ = np.squeeze(state_)
                print("squ", state_)
                select_train_, select_label_, select_num2 = actor.sample(tf.convert_to_tensor(state_),
                                                                                         X_train_t, y_train_t)

                td_loss, total_loss1, total_loss2 = parser.predict_self(select_train, select_label, select_train_,
                                                                        select_label_, X_dev)
                td_sum+=GAMMA**bag_id*td_loss

                print("td_loss", td_loss)
                tderror_save["tderror_save"+"_bag_id_"+str(bag_id)]=td_loss
                select_num2_save["select_num2"+"_bag_id_"+str(bag_id)]=select_num2

                # td_loss=0.05
                #with globals()['g_' + str(bag_id)].as_default():
                #    globals()['actor__' + str(bag_id)].learn(W_data_t_3d, td_loss)
                state = state_
                select_train = select_train_
                select_label = select_label_
                test4bag=True
                if test4bag:
                    conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
                    testpath = os.path.join(options.output,
                                    'test_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
                    utils.write_conll(testpath, parser.Predict(path_test))
                    parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))
                    print("predicting....")
                    if not conllu:
                       # os.system('perl src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
                       os.system('perl ./utils/eval.pl -g ' + path_test + ' -s ' + testpath + ' > ' + 'temp' + '.txt')
                    else:
                       # os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                       os.system(
                    'python ./utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + path_test + ' ' + testpath + ' > ' + 'temp' + '.txt')
                    with open('./temp.txt','r') as f:
                         iii=0
                         for line in f.readlines():
                             iii+=1
                             if iii>4:
                                break
                             else:
                                if iii==1:
                                   line_las=line.replace('\n','')
                                   line_las=line_las.replace('%','')
                                   line_las=line_las.split('=')
                                   line_las=float(line_las[1])
                                   acc=line_las
                                   if best_acc<acc:
                                      best_acc=acc
                             print(line.replace('\n',''))
                    print("best_las",best_acc) 
					   
					   
                    # saver.save(sess, best_path_actor)
                print("bag_id", bag_id, "finish actor pretrain_epoch {} ...".format(actor_epoch), "acc_now", acc,
                      "best_acc", best_acc)
            print("updating SDG...")
            td_sum_save["td_sum_save"+"_bag_id_"+str(bag_id)]=td_sum

            actor.learn(W_data_t_3d, td_sum)

            print("critic predicting...")
            conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
            testpath = os.path.join(options.output,
                                    'test_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(testpath, parser.Predict(path_test))
            parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))
            print("predicting....")
            if not conllu:
                # os.system('perl src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
                os.system('perl ./utils/eval.pl -g ' + path_test + ' -s ' + testpath + ' > ' + 'temp' + '.txt')
            else:
                # os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                os.system(
                    'python ./utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + path_test + ' ' + testpath + ' > ' + 'temp' + '.txt')

            with open('./temp.txt', 'r') as f:
                iii = 0
                for line in f.readlines():
                    iii += 1
                    if iii > 4:
                        break
                    else:
                        print(line.replace('\n', ''))

        print("end")

