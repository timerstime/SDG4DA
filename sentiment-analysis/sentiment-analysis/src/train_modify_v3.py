# -*- coding:utf-8 -*-
#! /usr/bin/env python
#--date:20180121   --
#--author:hubertzou --
#daliy log:
'''
#20180123:
增加了policy的shuffle部分
******todo*****：
1.取出所有的参数,将之前的参数也纳入考虑
2.本文的embedding需要重新做。



'''
import time
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_utils
from text_cnn import TextCNN
from tensorflow.contrib import learn
from RL_brain import PolicyGradient
from Data_select import Data_select
import math
#词表示是固定的

# Parameters
# Parameters
# ==================================================
tf.flags.DEFINE_string("debug_flag",'mac', "")
# Data loading params
#tf.flags.DEFINE_float("test_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("positive_data_file_train", "./data/rt-polaritydata/rt-polarity.pos.train", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file_train", "./data/rt-polaritydata/rt-polarity.neg.train", "Data source for the negative data.")

tf.flags.DEFINE_string("positive_data_file_test", "./data/rt-polaritydata/rt-polarity.pos.test", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file_test", "./data/rt-polaritydata/rt-polarity.neg.test", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters

tf.flags.DEFINE_integer("num_non_layer_features",384, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("batch_size",64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("bag_size_all",5331, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("bag_size",5331, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("episodes", 300, "episodes Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs",200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#add
tf.flags.DEFINE_integer("num_classes",2, "classes Size (default: 2)")
tf.flags.DEFINE_integer("min_frequency",0, "min_frequency Size (default: 2)")
tf.flags.DEFINE_string("dataset_name",'rt-polaritydata', "dataset_name")

tf.flags.DEFINE_integer("off_pretrain_policy",10, "")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#u以下用于配置在mac上跑或者在ubuntu跑
if FLAGS.debug_flag=='mac':
    FLAGS.num_epochs=1
    FLAGS.bag_size_all=1024
    FLAGS.bag_size=64
    FLAGS.off_pretrain_policy=10

if FLAGS.debug_flag=='ubuntu_train':
    FLAGS.num_epochs=10
    FLAGS.bag_size_all=4265
    FLAGS.bag_size=4265
    FLAGS.off_pretrain_policy=50

if FLAGS.debug_flag=='ubuntu_test':
    FLAGS.num_epochs=5
    FLAGS.bag_size_all=4265
    FLAGS.bag_size=4265
    FLAGS.off_pretrain_policy=10


#print("\nParameters:")

for attr, value in sorted(FLAGS.__flags.items()):
    pass
    #print("{}={}".format(attr.upper(), value))
#print("")


# ==================================================
#以下用于加载训练集的全部数据,为了建立词汇表,这样做的目的是为了方便训练，因为可能训练的时候不能加载全部数据
x_text_all, y_all = data_utils.load_data_and_labels(FLAGS.positive_data_file_train, FLAGS.negative_data_file_train, FLAGS.bag_size_all)
#max_document_length = 53  #暂时固定起来，方便分析 #max(max([len(x.split(" ")) for x in x_text_pos]),max([len(x.split(" ")) for x in x_text_pos]))
max_document_length =max([len(x.split(" ")) for x in x_text_all])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=FLAGS.min_frequency)
vocab_processor.fit(x_text_all)

x_all = np.array(list(vocab_processor.fit_transform(x_text_all)))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_all)))
x_shuffled = x_all[shuffle_indices]
y_shuffled = y_all[shuffle_indices]

# Split train/test set
# 取出所有数据，预训练cnn模型。
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_all)))

x_train= x_shuffled[:]
y_train= y_shuffled[:]


# ==================================================
#以下用于加载训练集的全部数据,为了建立词汇表,这样做的目的是为了方便训练，因为可能训练的时候不能加载全部数据
x_text_all_test, y_all_test = data_utils.load_data_and_labels(FLAGS.positive_data_file_test, FLAGS.negative_data_file_test, FLAGS.bag_size_all)
#max_document_length = 53  #暂时固定起来，方便分析 #max(max([len(x.split(" ")) for x in x_text_pos]),max([len(x.split(" ")) for x in x_text_pos]))
x_all_test = np.array(list(vocab_processor.fit_transform(x_text_all_test)))
x_dev= x_all_test
y_dev= y_all_test
# Split train/test set
# 取出所有数据，预训练cnn模型。
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_all)))


g_cnn=tf.Graph()#计算cnn的图
createVar = globals()

for item in range(FLAGS.num_classes):

        createVar['Data_select_' + str(item)] =Data_select(
        x_=None,
        x_text_=None,
        y_=None,
        file_path_=None,
        x_new_=None,
        y_new_=None,
        select_list_=None,
        select_list_new_=None,
        x_text_all_=None,
        non_layer_=None,
        )

        createVar['g_' + str(item)] =tf.Graph()
        createVar['all_new_x_' + str(item)] =None
        createVar['all_new_y_' + str(item)] = None
        with globals()['g_' + str(item)].as_default():
            createVar['RL_' + str(item)] = PolicyGradient(
                n_actions=2,  # np.ones((x_neg.shape[0],), dtype=int),
                n_features=FLAGS.num_non_layer_features,
                learning_rate=0.02,
                reward_decay=0.99,
                # output_graph=True,
            )


if(FLAGS.dataset_name=='rt-polaritydata'):
    Data_select_0.file_path_=FLAGS.negative_data_file_train
    Data_select_1.file_path_=FLAGS.positive_data_file_train

for item in range(FLAGS.num_classes):
    Data_select=globals()['Data_select_' + str(item)]
    Data_select.x_text_,Data_select.y_= data_utils.load_data_and_labels_modify_v2(Data_select.file_path_, FLAGS.bag_size, FLAGS.num_classes, item)
    Data_select.x_=np.array(list(vocab_processor.fit_transform(Data_select.x_text_)))
    #print(Data_select.x_.shape)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))



with g_cnn.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=Data_select_0.x_.shape[1],
            num_classes=FLAGS.num_classes,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "1"))
        os.system('rm -rf '+out_dir+'*')
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        #look_up
        array2lookup_int = tf.placeholder(tf.int64, [None, None], name="idx_lookup_int")
        array2lookup = tf.placeholder(tf.float64, [None, None], name="idx_lookup")
        idx_lookup = tf.placeholder(tf.int64, [None, ], name="idx_lookup")
        looup_op = tf.nn.embedding_lookup(array2lookup, idx_lookup)
        looup_op_int = tf.nn.embedding_lookup(array2lookup_int, idx_lookup)

        #use as:
        #array2lookup_value=np.random.random([64,348])
        #idx_lookup_value=np.arange(64)
        #sess.run(looup_op, feed_dict={array2lookup: array2lookup_value, idx_lookup: idx_lookup_value})

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss,prob,nonlayer,accuracy = sess.run(
                [train_op, global_step, train_summary_op,cnn.loss,cnn.prob,cnn.nonlayer,cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #print(nonlayer.shape)
            #print(prob)
            #print(sess.run())
            train_summary_writer.add_summary(summaries, step)
            return nonlayer,prob

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, prob,nonlayer,accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss,cnn.prob,cnn.nonlayer, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            return nonlayer, prob,accuracy

        # Generate batches
        batches = data_utils.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            #print(x_batch, y_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                n1,p1,acc_ori=dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


        #初始化,得到句子的表示
        listB=[]
        for item in range(FLAGS.num_classes):

            Data_select = globals()['Data_select_' + str(item)]

            Data_select.setlist()  #初始值为全选

            Data_select.x_,Data_select.y_,Data_select.select_list_len= data_utils.choose_From_Ori(Data_select.x_, Data_select.y_, Data_select.select_list_)

            batches = data_utils.batch_iter(list(zip(Data_select.x_, Data_select.y_)), Data_select.select_list_len, 1, Data_select.select_list_)

            print('the first run batch. classes:',item,'info:')
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                Data_select.non_layer_save,Data_select.problity,acc2 = dev_step(x_batch, y_batch)
            print(Data_select.non_layer_save.shape)

            Data_select.non_layer_save_first=Data_select.non_layer_save
            Data_select.x_first = Data_select.x_


            listB.append(item)

        pretrain_policy_off=False
        for episode in range(FLAGS.episodes):
            time_start=time.time()

            if episode>FLAGS.off_pretrain_policy:
                pretrain_policy_off=True
            # random.shuffle(listB)  #按照论文，随机打乱要输入的B,这样貌似不对，因为shuffle没用
            for item_for_bag in listB:

                Data_select = globals()['Data_select_' + str(item_for_bag)]
                RL_select = globals()['RL_' + str(item_for_bag)]

                #以下代码实现shuffle：non_layer_save，x_ 因为都是同一个类标，所以y就不需要shuffle

                Data_select.idx_lookup_value=np.arange(Data_select.non_layer_save.shape[0])
                np.random.shuffle(Data_select.idx_lookup_value)

                Data_select.non_layer_save=sess.run(looup_op, feed_dict={array2lookup: Data_select.non_layer_save, idx_lookup: Data_select.idx_lookup_value})
                Data_select.x_=sess.run(looup_op_int, feed_dict={array2lookup_int: Data_select.x_, idx_lookup: Data_select.idx_lookup_value})

                #先清零
                Data_select.clearlistnew()

                i = 0
                observation =Data_select.non_layer_save[0]  # non_layer_neg[0]  # 开始前，先把第一句的状态作为初始状态

                for item_i in Data_select.x_:

                    action_i = RL_select.choose_action(observation)  # 句子的ai
                    reward = 0  # 根据论文，只要这个batch没跑完，则保持reward为0

                    Data_select.select_list_new_[i] = action_i  # 更新下次是否选这个句子
                    # 下面计算之前句子的平均值
                    non_layer_avg = np.zeros((FLAGS.num_non_layer_features,), dtype=np.float32)
                    non_layer_sum = np.zeros((FLAGS.num_non_layer_features,), dtype=np.float32)
                    k = 0
                    count = 0.0
                    for select_item in Data_select.select_list_new_:
                        if select_item == 1 and k<i:
                            count += 1.0
                            non_layer_sum = non_layer_sum + Data_select.non_layer_save[k]  # 求出和值
                        k += 1

                    if count != 0:
                        non_layer_avg = non_layer_sum / count
                    observation_ = non_layer_avg+Data_select.non_layer_save[i]  # 这里的新observation_应该是已选择句子的平均值？
                    RL_select.store_transition(observation, action_i, reward)
                    observation = observation_
                    i += 1


                # 根据选择的句子计算reward
                count_B_select = Data_select.select_list_new_.count(1)



                if (count_B_select == 0):
                    # 如果选择了0个句子，则选择所有句子，那只是计算奖励，还是用所有的句子去更新网络？
                    Data_select.setlistnew()

                #将本个batch中所选的句子全部挑出来，用于更新回合结束后的cnn网络
                #Data_select.new_select_x_=np.zeros([count_B_select,max_document_length])
                #Data_select.update_new_x_new_y()

                # 根据新学到的select_list_new 或者选择所有的数据 来选择需要送入cnn的句子
                Data_select.x_new_,Data_select.y_new_,Data_select.select_list_len=data_utils.choose_From_Ori(Data_select.x_,
                                                                                                             Data_select.y_,
                                                                                                             Data_select.select_list_new_)

                batches = data_utils.batch_iter(
                    list(zip(Data_select.x_new_ , Data_select.y_new_)), Data_select.select_list_len, 1,shuffle=False)

                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                with g_cnn.as_default():
                    nonlayer, prob,acc2 = dev_step(x_batch, y_batch)   #不更新，只是为了取出
                eposide_prob = []

                for list_i in prob.tolist():
                    eposide_prob.append(math.log(list_i[item_for_bag]))  # 因为neg对应标签为[1,0]
                reward = sum(eposide_prob) / len(eposide_prob)
                RL_select.store_transition(observation, action_i, reward)

                with globals()['g_' + str(item_for_bag)].as_default():   #为了确保安全，在不同的图下调用不同的计算
                    vt = RL_select.learn()

                time_end = time.time()
                print('episode', episode, 'classes', item_for_bag, ' select', count_B_select, 'time cost',
                      time_end - time_start, 'off_pretrain',
                      pretrain_policy_off)

            if pretrain_policy_off:

                #all_new_select_x存储所有的已选择的值

                all_new_select_x=np.concatenate((Data_select_0.x_new_,Data_select_1.x_new_),axis=0)
                all_new_select_y = np.concatenate((Data_select_0.y_new_, Data_select_1.y_new_), axis=0)
                '''
                for item_kk in range(len(listB)-1):

                    if item==0:
                        all_new_select_x=np.concatenate((globals()['Data_select_' + str(item)].x_new_,globals()['Data_select_' + str(item+1)].y_new_),axis=0)
                        all_new_select_y=np.concatenate((globals()['Data_select_' + str(item)].x_new_,globals()['Data_select_' + str(item+1)].y_new_),axis=0)
                    else:
                        all_new_select_x = np.concatenate((globals()['Data_select_' + str(item)].x_new_,
                                                           globals()['Data_select_' + str(item + 1)].x_new_),axis=0)
                        all_new_select_y = np.concatenate((globals()['Data_select_' + str(item)].y_new_,
                                                           globals()['Data_select_' + str(item + 1)].y_new_),axis=0)
                '''
                #疑问：这里使用所有的数据一次喂入还是按小批量的batch_size喂入？
                batches = data_utils.batch_iter(
                    list(zip(all_new_select_x,all_new_select_y)),all_new_select_x.shape[0], 1,shuffle=True)

                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    # print(x_batch, y_batch)
                    train_step(x_batch, y_batch)

                if episode % 2 == 0:#在测试集上检验效果
                    print("\nEvaluation:")
                    n2,p2,acc2=dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("acc_ori",acc_ori,"new_select_acc",acc2)

