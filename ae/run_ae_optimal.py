import torch
import numpy as np
import os
import pandas as pd

from sklearn.metrics import roc_auc_score
import time
from model.model import  ParameterAE
from torch.optim import Adam
from model.utils_ae import dataLoading,weights_init_normal,cal_entropy,set_seed,get_ae_hps


def naive_train(x, y, act_func='relu', dropout=0.2, h_dim=64, num_layer=1, lr=0.001, epoch=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)

    model = ParameterAE(X.size(1), h_dim=h_dim, num_layer=num_layer, dropuout=dropout, first_layer_act=act_func).to(device)
    model.apply(weights_init_normal)
    opt = Adam(model.parameters(), lr=lr)

    from tqdm import tqdm

    # shuffle x index
    random_index = np.arange(X.size(0))
    np.random.shuffle(random_index)
    train_X = X[random_index]

    start = time.time()
    AUC = []
    for _ in tqdm(range(epoch)):
        for i in range(0, X.size(0), batch_size):
            model.train()
            batch_x = train_X[i:i + batch_size]
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
                AUC.append(auc)

    end = time.time()
    train_time = end - start

    

    return np.max(AUC), train_time



if __name__ == '__main__':
    template_model_name = 'ae'

    batch_size = 256
    seeds = [0,1,2]

    act_funcs, dropouts, h_dims, num_layers, lrs, epochs = get_ae_hps() # get all the hp configs

    for seed in seeds:
        g = os.walk(r"../data/data-for-primary")

        for path, dir_list, file_list in g:
            for file_name in file_list:
                AUC = []
                print(file_name)
                # if os.path.exists(f'./ae-nai-{file_name}.csv'):
                #     continue
                data_path = os.path.join(path, file_name)
                x, y = dataLoading(data_path)

                #  run all the hp configs
                for act in act_funcs:
                    for dropout in dropouts:
                        for h_dim in h_dims:
                            for num_layer in num_layers:
                                for lr in lrs:
                                    for epoch in epochs:
                                        print(file_name)
                                        print(f'hp: {act}-{dropout}-{h_dim}-{num_layer}-{lr}-{epoch}')
                                        
                                        set_seed(seed)
                                        auc,_ = naive_train(x, y, act, dropout, h_dim, num_layer, lr, epoch)

                                        AUC.append(auc)
                pd.DataFrame({'AUC':AUC}).to_csv(f'./ae-optimal-{file_name}-{seed}.csv')
                                        
    