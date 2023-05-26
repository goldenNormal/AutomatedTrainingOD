import json
import torch
import numpy as np
import os
import pandas as pd
import time
from sklearn.metrics import  roc_auc_score

from model.model import  ParameterAE
from torch.optim import Adam

from model.utils_ae import dataLoading,weights_init_normal,cal_entropy,set_seed,get_ae_hps,get_default_hp


def run_en(x, y):
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
    ES = ModelEntropyEarlyStop(k=k,R_down=r_down)

    N_eval = min(n_eval, X.shape[0])
    eval_index = np.random.choice(X.shape[0], N_eval, replace=False)
    x_eval = X[eval_index]

    start = time.time()
    isStop = False
   
    for _ in tqdm(range(epoch)):
        for batch_idx in range(0, X.size(0), batch_size):
            model.train()
            batch_x = train_X[batch_idx:batch_idx + batch_size]
            recon_score = model(batch_x)
            recon_loss = torch.mean(recon_score)
            loss = recon_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
        

    
            with torch.no_grad():
                model.eval()
                eval_x = x_eval
                eval_score = model(eval_x).cpu().detach().numpy()

                e = cal_entropy(eval_score)
                isStop = ES.step(e,model)
                if isStop:
                    break
        if isStop:
            break

    end = time.time()
    train_time = end - start

    with torch.no_grad():
        model=ES.getBestModel()
        model.eval()
        numpy_recon_score = model(X).cpu().detach().numpy()

    

    # epoch_of_max = int(np.argmax(Auc) // (len(Auc) / epoch))
    return roc_auc_score(y,numpy_recon_score), train_time


def run_naive(x, y):
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
    # train_y = y[random_index]

    start = time.time()
    for _ in tqdm(range(epoch)):
        
        
        for batch_idx in range(0, X.size(0), batch_size):
            model.train()
            batch_x = train_X[batch_idx:batch_idx + batch_size]
            recon_score = model(batch_x)
            recon_loss = torch.mean(recon_score)
            loss = recon_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
    end = time.time()
    train_time = end - start

    with torch.no_grad():
        model.eval()
        numpy_recon_score = model(X).cpu().detach().numpy()

    # epoch_of_max = int(np.argmax(Auc) // (len(Auc) / epoch))
    return roc_auc_score(y,numpy_recon_score), train_time

from model.outlier_generate import generate_realistic_synthetic


def dataLoadingInject(path, outlier_type, ratio):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    y = data['y']
    print(x.shape, y.shape)
    if outlier_type is not None:
        print("Injecting outliers...")
        x, y = generate_realistic_synthetic(
        X=x,
        y=y,
        realistic_synthetic_mode=outlier_type,
        ratio=ratio
    )
    return x, y



if __name__ == '__main__':
    template_model_name = 'ae'

    
    n_eval, batch_size, act_func, dropout, h_dim, num_layer, lr, epoch, k, r_down = get_default_hp()

    types = ['global','cluster','local']
    ratio = [0.1,0.4]

    
    for ot in types:
        for r in ratio:
            if os.path.exists(f'./inject-{r}-{ot}-ae.csv'):
                continue

            g = os.walk(f"../data/data-for-inject/")
            
            EN_AUC = []
            EN_Time = []
            Nai_Time = []
            NAI_AUC = []
            Dataset = []
            Seed = []
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    
                    print(file_name)
                    data_name = file_name.split('.')[0].split('_')[1]
                    for seed in range(3):
                        set_seed(seed)
                        

                        x,y = dataLoadingInject(os.path.join(path, file_name),outlier_type=ot,ratio=r)
                        
                        set_seed(seed)
                        en_auc,en_time = run_en(x,y)
                        set_seed(seed)
                        nai_auc,nai_time = run_naive(x,y)
                        EN_AUC.append(en_auc)
                        EN_Time.append(en_time)
                        Nai_Time.append(nai_time)
                        NAI_AUC.append(nai_auc)
                        Dataset.append(data_name)
                        Seed.append(seed)

                  
            pd.DataFrame({'dataset':Dataset,'en_auc':EN_AUC,'nai_auc':NAI_AUC,'en_time':EN_Time,'nai_time':Nai_Time,'seed':Seed}).to_csv(f'./results-inject-{r}-{ot}-ae.csv',index=False)