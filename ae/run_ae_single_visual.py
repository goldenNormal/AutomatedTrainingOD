import json
import torch
import numpy as np
import os
import pandas as pd

from sklearn.metrics import roc_auc_score

from model.model import ParameterAE
from torch.optim import Adam
from EntropyEarlyStop import getStopOffline

from model.utils_ae import cal_entropy, set_seed, dataLoading, weights_init_normal


def template_train_eval(x, y, act_func='relu', dropout=0.2, h_dim=64, num_layer=1, lr=0.001, epoch=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)

    model = ParameterAE(X.size(1), h_dim=h_dim, num_layer=num_layer, dropuout=dropout, first_layer_act=act_func).to(device)
    model.apply(weights_init_normal)
    opt = Adam(model.parameters(), lr=lr)

    from tqdm import tqdm

    Auc = []
    En = []

    

    for _ in tqdm(range(epoch)):
        for i in range(0, X.size(0), batch_size):
            model.train()
            batch_x = X[i:i + batch_size]
            recon_score = model(batch_x)
            recon_loss = torch.mean(recon_score)
            loss = recon_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

         
            with torch.no_grad():
                model.eval()
                numpy_recon_score = model(X).cpu().detach().numpy()

                auc = roc_auc_score(y, numpy_recon_score)
                Auc.append(auc)
                e = cal_entropy(numpy_recon_score)
                En.append(e)

    return Auc, En


def run_one(path, file_name, act_func='relu', dropout=0.2, h_dim=64, num_layer=1, lr=0.001, training_epoch=100):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x, y = dataLoading(data_path)

    return template_train_eval(x, y, act_func, dropout, h_dim, num_layer, lr, training_epoch)


if __name__ == '__main__':
    template_model_name = 'ae'
    
    batch_size = 256
    seed = 0

    act_func = 'relu'
    dropout = 0.0
    h_dim =  64
    num_layer = 1  
    lr = 0.001
    epoch = 100

    k = 100
    R_down = 0.1


    g = os.walk(r"../data/data-for-primary")


    selected_datasets = [
        '18_Ionosphere.npz',
        '20_letter.npz',
        '40_vowels.npz',
        'MNIST_3.npz',
        'MNIST_5.npz'
    ]

    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name not in selected_datasets:
                continue
            
            print(file_name)

            
            set_seed(seed)
            print(file_name)

            AUC, En = run_one(path, file_name, act_func, dropout, h_dim, num_layer, lr, epoch)
            import matplotlib.pyplot as plt
            
            plt.title(file_name)
            plt.subplot(2,1,1)
            plt.plot(AUC,label = 'AUC')
            plt.subplot(2,1,2)
            plt.plot(En,label = 'EN')
            
            select_iter = getStopOffline(En, k, R_down)
            plt.vlines([select_iter],np.max(En),np.min(En),color='red',linestyles='dashed',label='select_iter')

            plt.legend()
        
            if not os.path.exists('./training-img/'):
                os.mkdir('./training-img')
            plt.savefig(f'./training-img/{file_name}.png')
            # plt.show()
            plt.clf()
               
    

