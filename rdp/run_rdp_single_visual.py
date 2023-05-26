import datetime
import platform
import shutil
import os
import sys
from model.util import  random_list
import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from model.model import RDP_Model
import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
from model.util import random_list

import random
import torch
import numpy as np
def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False



def dataLoading(path, logfile=None):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    # 归一化
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    print(x.shape)
    y = data['y']
    return x,y

import numpy as np
import copy
def cal_entropy(score):
    score = score.reshape(-1)
    score = score/np.sum(score) # to possibility
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy

class RDPTree():
    def __init__(self,
                 t_id,
                 tree_depth,
                 filter_ratio=0.1):

        self.t_id = t_id
        self.tree_depth = tree_depth
        self.filter_ratio = filter_ratio
        self.thresh = []

    # include train and eval
    def training_process(self,
                         x,
                         labels,
                         batch_size,
                         node_batch,
                         node_epoch,
                         eval_interval,
                         out_c,
                         USE_GPU,
                         LR,
                         save_path,
                         logfile=None,
                         dropout_r=0.1,
                         svm_flag=False,
                         ):
        if svm_flag:
            x_ori = x.toarray()
        else:
            x_ori = x
        labels_ori = labels
        x_level = np.zeros(x_ori.shape[0])


        # form x and labels
        keep_pos = np.where(x_level == 0)
        x = x_ori[keep_pos]
        labels = labels_ori[keep_pos]
        group_num = int(x.shape[0] / batch_size) + 1
        batch_x = np.array_split(x, group_num)
        model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                          LR=LR, logfile=logfile, dropout_r=dropout_r)
        
        best_auc = best_epoch = 0

        AUC = []
        EN = []

        # sample_interval = node_epoch / 300
        # update_interval = 15
        N_eval = min(1024, x_ori.shape[0])
        eval_index = np.random.choice(x_ori.shape[0], N_eval, replace=False)
        x_eval = x_ori[eval_index]

        update_cnt = 0
        Epoch = []
        for epoch in tqdm(range(0, node_epoch)):

            for batch_i in range(node_batch):
                
                random_pos = random_list(0, x.shape[0] - 1, batch_size)
                batch_data = x[random_pos]
                gap_loss = model.train_model(batch_data, epoch)

                update_cnt +=1
                if  True:
                    scores = model.eval_model(x_ori)
                    auc = roc_auc_score(labels_ori,scores)
                    # print(scores.shape)

                    AUC.append(auc)
                    Epoch.append(epoch)
                    EN.append(cal_entropy(model.eval_model(x_eval)))
                
        max_auc_index = np.argmax(AUC)
        epoch_of_max = Epoch[max_auc_index]

        return AUC, epoch_of_max,EN

def main(data_path):
    # global random_size

    x_ori, labels_ori = dataLoading(data_path, logfile)
    data_size = labels_ori.size
    print(x_ori.shape)
    # build forest
    forest = []
    for i in range(forest_Tnum):
        forest.append(RDPTree(t_id=i+1,
                              tree_depth=tree_depth,
                              filter_ratio=filter_ratio,
                              ))
    x = x_ori

    labels = labels_ori


    return forest[0].training_process(
        x=x,
        labels=labels,
        batch_size=batch_size,
        node_batch=node_batch,
        node_epoch=node_epoch,
        eval_interval=eval_interval,
        out_c=out_c,
        USE_GPU=USE_GPU,
        LR=LR,
        save_path=save_path,
        logfile=logfile,
        dropout_r=dropout_r,

    )



def run_one(path,file_name):

    data_path = os.path.join(path, file_name)

    return main(data_path)
import argparse
if __name__ == '__main__':
    template_model_name = 'rdp'
    
    g = os.walk(r"../data/data-for-primary")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--outc', type=int, default=50)
    # parser.add_argument('--neval', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--filter', type=float, default=0.05)
    parser.add_argument('--epoch', type=int, default=100)
 
    args = parser.parse_args()
    
    save_path = "save_model/"
    log_path = "logs/log.log"
    logfile = None
    node_batch = 30
    node_epoch = args.epoch  # epoch for a node training
    eval_interval = 24
    batch_size = 256
    out_c = args.outc
    USE_GPU = True
    LR =args.lr
    tree_depth = 1 # boost
    forest_Tnum = 1 # boost
    filter_ratio = args.filter  # filter those with high anomaly scores
    dropout_r = args.dropout
  


    is_batch_replace = True
    is_eval = False
    test_1l_only = True

    set_seed(0)

    selected_datasets = [
        '18_Ionosphere.npz',
        '20_letter.npz',
        '40_vowels.npz',
        'MNIST_3.npz',
        'MNIST_5.npz'
    ]


    cnt = 0
    Dict = dict()
    for path,dir_list,file_list in g:
       
        for file_name in file_list:
            if file_name not in selected_datasets:
                continue
            print(file_name)
            
            data_path = os.path.join(path, file_name)
            x_ori, labels_ori = dataLoading(data_path, logfile)


            AUC,epoch_of_max,EN = run_one(path,file_name)
            print(epoch_of_max)
            
            import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            plt.title(file_name)
            plt.subplot(2,1,1)
            plt.plot(AUC,label = 'AUC')
            plt.subplot(2,1,2)
            plt.plot(EN,label = 'EN')
            
            plt.legend()
        
            if not os.path.exists('./training-img/'):
                os.mkdir('./training-img')
            plt.savefig(f'./training-img/{file_name}-dropout{dropout_r}-outc{out_c}-lr{LR}-filter{filter_ratio}-epoch{node_epoch}.png')
            # plt.show()
            plt.clf()

