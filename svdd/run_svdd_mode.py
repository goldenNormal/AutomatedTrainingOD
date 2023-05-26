import json
import time
import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.metrics import roc_auc_score

from torch.optim import Adam

import os
import sys

import random
import torch
import numpy as np

from model.utils_for_svdd import dataLoading, set_seed,  get_svdd_hps

sys.path.append(os.path.dirname(sys.path[0]))

from model.DeepSVDDmodel import ConvDeepSVDD
from sklearn.preprocessing import MinMaxScaler


def run_one(path, file_name, conv_dim, fc_dim=32, relu_slope=0.1, pre_iteration=100, pre_lr=1e-4, iteration=250,
            lr=1e-4, weight_decay=1e-6):
    print('dataset : ', file_name)
    filepath = os.path.join(path, file_name)
    x, y = dataLoading(filepath)

    

    model = ConvDeepSVDD(
        conv_dim=conv_dim,
        fc_dim=fc_dim,
        relu_slope=relu_slope,
        pre_train=False,
        pre_train_weight_decay=weight_decay,
        train_weight_decay=weight_decay,
        pre_train_epochs=pre_iteration,
        pre_train_lr=pre_lr,
        train_epochs=iteration,
        train_lr=lr,
        dataset= 'mnist'
    )
    if args.train == 'naive':
        return model.nai_train(x, y, 'mnist')
    elif args.train == 'entropy':
         return model.en_train(x, y, 'mnist')
    elif args.train == 'optimal':
         return model.opt_train(x, y, 'mnist')
    else:
         raise ValueError('train mode not supported')


if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='naive') # naive, entropy, optimal
    args = parser.parse_args()

    seeds = [0,1,2]
    template_model_name = 'svdd'
    conv_dims, fc_dims, relu_slopes, iterations, lrs, weight_decays = get_svdd_hps()

    selected_datasets = [
        'MNIST_3.npz',
        'MNIST_5.npz'
    ]

    for seed in seeds:
        g = os.walk(f"../data/data-for-primary")
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name not in selected_datasets:
                        continue
                AUC = []
                RT = []
                print(file_name)
                for conv_dim in conv_dims:
                    for fc_dim in fc_dims:
                        for relu_slope in relu_slopes:
                                    for iteration in iterations:
                                        for lr in lrs:
                                            for weight_decay in weight_decays:
                                
                                                set_seed(seed)
                                                
                                                print('hp:',
                                                    [conv_dim, fc_dim, relu_slope, iteration,
                                                        lr, weight_decay])
                                                auc,rt,outlier_score = run_one(path, file_name, conv_dim, fc_dim,
                                                                            relu_slope, 0, 0,
                                                                            iteration, lr, weight_decay)
                                                AUC.append(auc)
                                                RT.append(rt)
                                                
                                                if args.train == 'naive':
                                                    mode_txt = 'nai'
                                                    if not os.path.exists(f'../eval_uoms/prediction/{template_model_name}'):
                                                        os.makedirs(f'../eval_uoms/prediction/{template_model_name}')
                                                    if not os.path.exists(f'../eval_uoms/prediction/{template_model_name}/{file_name}'):
                                                        os.makedirs(f'../eval_uoms/prediction/{template_model_name}/{file_name}')
                                                    np.savetxt(f'../eval_uoms/prediction/{template_model_name}/{file_name}/{conv_dim}-{fc_dim}-{relu_slope}-{iteration}-{lr}-{weight_decay}-{seed}.txt', outlier_score)
                                                elif args.train == 'entropy':
                                                    mode_txt = 'en'
                                                elif args.train == 'optimal':
                                                    mode_txt = 'opt'
                                                else:
                                                    raise ValueError('train mode not supported')

                pd.DataFrame({'AUC':AUC,'Time':RT}).to_csv(f'./svdd-{mode_txt}-{file_name}-{seed}.csv',index=False)

