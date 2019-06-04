# -*- coding:utf-8 -*-
#方法：使用预训练的加载下数据就好！
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
iris = load_iris()
data=iris.data
print iris.data
print "####"
print iris.target
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
X_pca = PCA().fit_transform(iris.data)
c=iris.target
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.show()

'''


root_save=["./target_dvd_renyi","./target_dvd_js","./target_dvd_loss","./target_dvd_distance"]
root_save_color=['r','y','b','g']
marker_list=['|','o','x','v']
line_style=[':','--','-.','-']
label_list=['Renyi divergence','Jensen Shannon divergence','Guidance data\'s loss','Euclidean distance']
ii=0

for root in root_save:
    if root!="./target_dvd_renyi":
        break
    path=os.listdir(root)
    if ".DS_Store" in path:
       path.remove(".DS_Store")
    path_int=[]
    for item in path:
        path_int.append(int(item))
    path_int.sort()

    path_dict=[]
    for item in path_int:
        path_dict.append(os.path.join(root,str(item)))
    print("root",root)

    acc_save_0_all=[]
    best_acc_save_0=[]
    jj = 0

    begin=0
    end=14
    buxuan=0
    random=2
    number=200
    for item in path_dict:

        if jj==buxuan:

            print(item)
            output_pkl = open(item, 'rb')
            dict_load_1=pickle.load(output_pkl)
            data_train_0=dict_load_1['w_data_t_save']['W_data_t_bag_id_0']

            data_train_1 = dict_load_1['w_data_t_save']['W_data_t_bag_id_1']

            data_train_2 = dict_load_1['w_data_t_save']['W_data_t_bag_id_2']

            data_train_3 = dict_load_1['w_data_t_save']['W_data_t_bag_id_3']

            data_train_4 = dict_load_1['w_data_t_save']['W_data_t_bag_id_4']

            data_train_5 = dict_load_1['w_data_t_save']['W_data_t_bag_id_5']



            #使用一个来自0的数据
            #if jj == begin:
                #use_data_train_0=data_train_0[:500]
                #use_data_train_1=data_train_1[300:800]
                #use_data_train_2 = data_train_0[500:1000]
                #use_data_train_3 = data_train_2[0:500]
            if jj == buxuan:
                use_data_train_0=data_train_0[:400]
                use_data_train_1=data_train_1[300:700]
                use_data_train_2 = data_train_2[600:1000]
                use_data_train_3 = data_train_5[0:400]

            #if jj==end:
                #use_data_train_0 = data_train_0[:200]
                #use_data_train_1 = data_train_1[300:350]
                #use_data_train_2 = data_train_0[500:750]
                #use_data_train_3 = data_train_2[0:450]


            #使用源域数据0
            c_train_0 = ['orange'] * use_data_train_0.shape[0]
            c_train_0 = np.array(c_train_0)

            # 使用源域数据0
            c_train_2 = ['pink'] * use_data_train_1.shape[0]
            c_train_2 = np.array(c_train_2)

            # 使用源域数据0
            c_train_3 = ['gray'] * use_data_train_2.shape[0]
            c_train_3 = np.array(c_train_3)



            #使用源域数据1
            c_train_1 = ['yellow'] * use_data_train_3.shape[0]
            c_train_1 = np.array(c_train_1)


            # 使用目标域数据
            data_dev=dict_load_1['x_data_t_dev_save']['X_data_t_dev_bag_id_2']
            data_dev = data_dev[:300]

            c_dev= ['red'] * data_dev.shape[0]
            c_dev = np.array(c_dev)

                        #使用源域数据0
            m_train_0 = ['o'] * use_data_train_0.shape[0]
            m_train_0 = np.array(m_train_0)

            # 使用源域数据0
            m_train_2 = ['v'] * use_data_train_1.shape[0]
            m_train_2 = np.array(m_train_2)

            # 使用源域数据0
            m_train_3 = ['s'] * use_data_train_2.shape[0]
            m_train_3 = np.array(m_train_3)



            #使用源域数据1
            m_train_1 = ['x'] * use_data_train_3.shape[0]
            m_train_1 = np.array(m_train_1)

            m_dev= ['+'] * data_dev.shape[0]
            m_dev = np.array(m_dev)


            data_all=np.concatenate((use_data_train_0,use_data_train_1,use_data_train_2,use_data_train_3,data_dev),axis=0)
            c_all=np.concatenate((c_train_0,c_train_1,c_train_2,c_train_3,c_dev),axis=0)
            m_all=np.concatenate((m_train_0,m_train_1,m_train_2,m_train_3,m_dev),axis=0)
            #marker_all=np.concatenate((marker_train,marker_dev),axis=0)

            X_tsne = TSNE(learning_rate=50).fit_transform(data_all)

            #if jj==begin:
                #plt.figure(figsize=(5, 5))
                #plt.xticks([])
                #plt.yticks([])
                #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_all,label='the 0 epoch')
                ##plt.legend(loc='upper left')
                #plt.savefig('./savepdf/' + label_list[ii] + "_plot_tsne_" + "begin" + '.pdf')
                #plt.show()

            #if jj==end:
                #plt.figure(figsize=(5, 5))
                #plt.xticks([])
                #plt.yticks([])
                #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_all, label='the 14 epoch')
                #plt.legend(loc='upper left')
                #plt.savefig('./savepdf/' + label_list[ii] + "_plot_tsne_" + "end" + '.pdf')
                #plt.show()

            if jj==buxuan:
                plt.figure(figsize=(5, 5))
                plt.xticks([])
                plt.yticks([])
                for i in range(np.size(data_all)):
                    plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=c_all[i], marker=m_all[i])
                #plt.legend(loc='upper left')
                plt.savefig('./savepdf/' + label_list[ii] + "_plot_tsne_" + "buxuan" + '.pdf')
                plt.show()

            output_pkl.close()
        jj += 1
    ii+=1

