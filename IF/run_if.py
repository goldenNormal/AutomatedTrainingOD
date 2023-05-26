import os
import sys
import time
import numpy as np

from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
# from sklearn.ensemble import IsolationForest
from pyod.models.iforest import IsolationForest

def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)
    

    
    AUCs = []
    APs = []

    IF = IsolationForest(n_estimators=100)
    start = time.time()
    IF.fit(x)
    end = time.time() -start
    score = (-1.0) * IF.decision_function(x)
    auc = roc_auc_score(y.reshape(-1),score.reshape(-1))
    ap =average_precision_score(y.reshape(-1),score.reshape(-1))
    AUCs.append(auc) 
    APs.append(ap)
    
    mean_auc = np.mean(AUCs)
    mean_ap = np.mean(APs)

    return mean_auc,mean_ap,end

from utils import dataLoading, set_seed

import argparse
if __name__ == '__main__':
    template_model_name = 'IF'
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--seed',type=int,default=0)

    args = parser.parse_args()
    for key,v in sorted(vars(args).items()):
        print(key,'=',v)
    # print()
    print('-'*50)

    set_seed(args.seed)

    try_times = 3

    for tt in range(try_times):
        g = os.walk(r"../data/data-for-primary")

        cnt = 0

        Dataset = []
        AUC = []
        Run_time = []
        AP = []

        for path,dir_list,file_list in g:
            for file_name in file_list:
                print(file_name)
                auc,ap,rt= run_one(path,file_name)
              
                Run_time.append(rt)
                AUC.append(auc)
                AP.append(ap)
               

                Dataset.append(file_name)
                print(cnt)

                cnt+=1


        df = pd.DataFrame({'Dataset':Dataset,
                           "auc":AUC,
                           'ap': AP,
                            'Run_time':Run_time
                           })
        df.to_csv(f'./{template_model_name}_results-{tt}.csv', index=False)
