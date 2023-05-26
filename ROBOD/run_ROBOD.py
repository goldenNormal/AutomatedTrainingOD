import torch
from sklearn.metrics import roc_auc_score
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd

from utils import dataLoading,set_seed

import numpy as np

def run_one(path,file_name):
    data_path = os.path.join(path, file_name)

    x,y = dataLoading(data_path)
    from Ensemble.ROBOD import LinearROBOD
    print('hp: ',first_act,dropout,lr,epoch)
    model = LinearROBOD(input_dim=x.shape[-1],device='cuda',epochs=epoch,lr=lr,dropout=dropout,first_activation=first_act)
    X = torch.FloatTensor(x)
    rt,mem,_ =model.fit(X)
    score =model.predict(X)
    auc = roc_auc_score(y,score)
    print('auc:',auc)
    return auc,rt,mem


if __name__ == '__main__':
    template_model_name = 'ROBOD'
    
    first_acts = ['relu','sigmoid']
    dropouts = [0,0.2]
    lrs = [0.005,0.001]
    epochs = [100 ,500]
 
    cnt = 0
    AUCs = []

    seeds = [0,1,2]

    for seed in seeds:
        for first_act in first_acts:
            for dropout in dropouts:
                for lr in lrs:
                    for epoch in epochs:
                        Auc = []
                        Dataset = []
                        Ap =[]
                        Mem = []
                        RunTime = []
                        g = os.walk(r"../data/data-for-primary")
                    
                        print(cnt)
                        for path,dir_list,file_list in g:
                            for file_name in file_list:
                                print(file_name)
                                set_seed(seed)

                                auc,rt,mem = run_one(path,file_name)

                                Auc.append(auc)
                                Dataset.append(file_name)
                                RunTime.append(rt)
                                Mem.append(mem)
                        


                        df = pd.DataFrame({'Dataset':Dataset,
                                        "Auc":Auc,
                                        "RunTime":RunTime,
                                        "Memory":Mem
                                        })
                        if not os.path.exists('./detail_results/'):
                            os.mkdir('./detail_results/')
                        df.to_csv(f'./detail_results/{template_model_name}-{first_act}-{dropout}-{lr}-{epoch}-{seed}.csv', index=False)
                        cnt +=1

                        AUCs.append(df['Auc'].values)
        AUCs = np.mean(np.stack(AUCs,axis=0),axis=0)
        pd.DataFrame({'Dataset':Dataset,"Auc":AUCs }).to_csv(f'ROBOD_results-{seed}.csv')
    
    
    