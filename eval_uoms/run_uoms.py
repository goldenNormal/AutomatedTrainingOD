import itertools
import numpy as np

def get_ae_hps():
    act_funcs = ['relu', 'sigmoid']
    dropouts = [0, 0.2]
    h_dims = [64,256]
    num_layers = [1, 2]  # This is just the layer of the encoder
    lrs = [0.005, 0.001]
    epochs = [100, 500]

    return act_funcs, dropouts, h_dims, num_layers, lrs, epochs

def get_rdp_hps():
    out_cs =  [25, 50,100]
    lrs = [0.1, 0.01]
    dropouts = [0.0, 0.1]
    filter_ratios = [0.0, 0.05, 0.1]
    epochs = [100, 500]
    return out_cs, lrs, dropouts, filter_ratios, epochs

def get_svdd_hps():
    conv_dims = [8, 16]
    fc_dims = [16, 64]
    relu_slopes = [0.1, 0.001]
    iterations = [
         100,250
    ]
    lrs = [1e-4, 1e-3]
    weight_decays = [
        1e-5,
        1e-6
    ]
    return conv_dims, fc_dims, relu_slopes, iterations, lrs, weight_decays


import pandas as pd
from sklearn.metrics import roc_auc_score
from uoms_implement import all_model_select_algorithms

def getY(data_name):
    
    return np.load('../data/data-for-primary/'+data_name+'')['y']

deepODs = ['ae','rdp','svdd']

datasets = [ ['18_Ionosphere.npz', 
        '20_letter.npz',
        '40_vowels.npz',
        'MNIST_3.npz',
        'MNIST_5.npz'],

        ['18_Ionosphere.npz', 
        '20_letter.npz',
        '40_vowels.npz',
       ],
        
        [
        'MNIST_3.npz',
        'MNIST_5.npz'] ]

params = None



XB = []
HIT = []
MC = []
OD = []
Dataset = []
Seed = []

for seed in [0,1,2]:

    for i in range(len(deepODs)):
        od_algorithm = deepODs[i]
        if od_algorithm == 'ae':
            params = get_ae_hps()
        elif od_algorithm == 'rdp':
            params = get_rdp_hps()
        elif od_algorithm == 'svdd':
            params = get_svdd_hps()
        else:
            params = None
        
        select_dataset = datasets[i]

        param_values = list(params)


        for data in select_dataset:
            label = getY(data)
        
            AUCs = []
            Scores = []
            

            # for seed in [0,1,2]:

            for values in itertools.product(*param_values):
                # dict_file = "-".join([f"{v}" for v in values]) + f"-{seed}.csv"
                outlier_file = "-".join([f"{v}" for v in values]) + f"-{seed}.txt"
                
                outlier_score = np.loadtxt(f'./prediction/{od_algorithm}/{data}/{outlier_file}')
                # en_score = pd.read_csv(f'{path}/{data}/{en_outlier_file}',header=None)
                
                Scores.append(outlier_score.reshape(-1))
                AUCs.append(roc_auc_score(label,outlier_score.reshape(-1)))

            print(f'{od_algorithm} on {data}')
            res = all_model_select_algorithms(Scores,AUCs,label)


        
            XB.append(res['XB'])
            HIT.append(res['Hit'])
            MC.append(res['MC'])
            OD.append(od_algorithm)
            Dataset.append(data)
            Seed.append(seed)
        
pd.DataFrame({'Dataset':Dataset,'OD':OD,'XB':XB,'HIT':HIT,'MC':MC,'seed':Seed}).to_csv(f'./uoms_results.csv', index=False)