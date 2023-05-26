
import os
import sys
from model.util import random_list
import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from model.model import RDP_Model
import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
from  model.util import random_list
import time
import numpy as np
from model.utils_for_rdp import dataLoading,set_seed,cal_entropy,get_rdp_hps


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

        update_cnt = 0

        
        AUC = []
        start = time.time()
        for epoch in tqdm(range(0, node_epoch)):

            for batch_i in range(node_batch):
                random_pos = random_list(0, x.shape[0] - 1, batch_size)
                batch_data = x[random_pos]
                gap_loss = model.train_model(batch_data, epoch)

                update_cnt += 1

                with torch.no_grad():
                    scores = model.eval_model(x_ori)
                    AUC.append(roc_auc_score(labels_ori, scores))


        end = time.time()
        train_time = end - start


        return np.max(AUC) ,train_time



import pandas as pd



if __name__ == '__main__':
    template_model_name = 'rdp'

    g = os.walk(r"../data/data-for-primary")

    node_batch = 30
    eval_interval = 24
    batch_size = 256
    seeds = [0,1,2]
    USE_GPU = True
  
    tree_depth = 1  # boost
    forest_Tnum = 1  # boost

    is_batch_replace = True
    is_eval = False
    test_1l_only = True

    out_cs, lrs, dropouts, filter_ratios, epochs = get_rdp_hps()

    selected_datasets = [
        '18_Ionosphere.npz',
        '20_letter.npz',
        '40_vowels.npz',
    ]

    for seed in seeds:


        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name not in selected_datasets:
                    continue


                print(file_name)

                AUC = []
                RT = []

                data_path = os.path.join(path, file_name)
                x,y = dataLoading(data_path)

                for out_c in out_cs:
                    for lr in lrs:
                        for dropout in dropouts:
                            for filter_ratio in filter_ratios:
                                for epoch in epochs:

                                    print(f'hp: {out_c}-{lr}-{dropout}-{filter_ratio}-{epoch}')
                                    set_seed(seed)
                                    tree = RDPTree(t_id=0,tree_depth=tree_depth, filter_ratio=filter_ratio)

                                    Auc,_ = tree.training_process(
                                            x=x,
                                            labels=y,
                                            batch_size=batch_size,
                                            node_batch=node_batch,
                                            node_epoch=epoch,
                                            eval_interval=eval_interval,
                                            out_c=out_c,
                                            USE_GPU=USE_GPU,
                                            LR=lr,
                                            save_path=None,
                                            logfile=None,
                                            dropout_r=dropout,
                                            )

                                    AUC.append(Auc)
                
                pd.DataFrame({'AUC': AUC}).to_csv(f'./rdp-optimal-{file_name}-{seed}.csv', index=False)



