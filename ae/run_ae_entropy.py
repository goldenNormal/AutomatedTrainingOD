import torch
import numpy as np
import os
import pandas as pd

from sklearn.metrics import  roc_auc_score
import time
from model.utils_ae import get_ae_hps
from model.model import  ParameterAE
from torch.optim import Adam
from model.utils_ae import cal_entropy, set_seed, dataLoading, weights_init_normal


def entropy_train(x, y, act_func='relu', dropout=0.2, h_dim=64, num_layer=1, lr=0.001, epoch=100):
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

    from EntropyEarlyStop import ModelEntropyEarlyStop
    ES = ModelEntropyEarlyStop(k=k_,R_down=0.1)

    N_eval = min(n_eval, X.shape[0])
    eval_index = np.random.choice(X.shape[0], N_eval, replace=False)
    x_eval = X[eval_index]

    start = time.time()
    isStop = False
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
                
                eval_score = model(x_eval).cpu().detach().numpy()
                e = cal_entropy(eval_score)
                isStop = ES.step(e,model)
                if isStop:
                    break
        if isStop:
            break

    end = time.time()
    train_time = end - start

    with torch.no_grad():
        model = ES.getBestModel()
        model.eval()
        numpy_recon_score = model(X).cpu().detach().numpy()


    return roc_auc_score(y,numpy_recon_score), train_time



if __name__ == '__main__':
    template_model_name = 'ae'

    n_eval = 1024
    batch_size = 256

    act_funcs, dropouts, h_dims, num_layers, lrs, epochs = get_ae_hps() # get all the hp configs

    R_down = 0.1
    


    k_for_dataset ={
        '18_Ionosphere.npz':25,
        '20_letter.npz':100,

        '40_vowels.npz':100,
        'MNIST_3.npz':50,
        'MNIST_5.npz':50
    }

    for seed in [0,1,2]:


        g = os.walk(r"../data/data-for-primary")

        for path, dir_list, file_list in g:
            for file_name in file_list:
                if file_name not in k_for_dataset.keys():
                    continue
                AUC = []
                Time = []
                print(file_name)
                k_ = k_for_dataset[file_name]


                data_path = os.path.join(path, file_name)
                x, y = dataLoading(data_path)

                for act in act_funcs:
                    for dropout in dropouts:
                        for h_dim in h_dims:
                            for num_layer in num_layers:
                                for lr in lrs:
                                    for epoch in epochs:
                                        print(f'hp: {act}-{dropout}-{h_dim}-{num_layer}-{lr}-{epoch}')
                                        
                                        set_seed(seed)
                                        auc,train_time = entropy_train(x,y, act, dropout, h_dim, num_layer, lr, epoch)
                                        AUC.append(auc)
                                        Time.append(train_time)
                pd.DataFrame({'AUC':AUC,'Time':Time}).to_csv(f'./ae-en-{file_name}-{seed}.csv')
                                        
    