import numpy as np
from time import time
from scipy.stats import spearmanr,kendalltau,rankdata
import math
from sklearn.metrics import roc_auc_score

## best cluster metrics :XB
def _xb(s, y):
    num_anomaly = np.sum(y==1)
    s_rank = np.sort(s)
    threshold = s_rank[-num_anomaly]
    c_normal, c_anomaly = np.mean(s_rank[:-num_anomaly]),np.mean(s_rank[-num_anomaly:])
    c = [c_anomaly if i >= threshold else c_normal for i in s]
    return sum((s-c)**2) / (len(s) * ((c_normal - c_anomaly) ** 2))

def XB(OD_scores,y):
    HPconfs = len(OD_scores)
    metrics = []
    for i in range(HPconfs):
        metrics.append( _xb(OD_scores[i],y))
    selected_index = np.argmax(metrics)

    return selected_index

def MC(OD_scores,_):
    HPconfs = len(OD_scores)
    rank_scores = [rankdata(score) for score in OD_scores]

    P = HPconfs  - 1
    MC_scores = []
    for i in range(HPconfs):
        corr_sum = 0
        weight = np.ones((HPconfs,))
        weight[i] = 0
        weight = weight/np.sum(weight)
        random_HPconfss = np.random.choice(np.arange(HPconfs),size=P,replace=False,p=weight).tolist()

        for j in random_HPconfss:
            i_s,j_s = rank_scores[i],rank_scores[j]
            corr = kendalltau(i_s, j_s)[0]
            corr_sum += corr
        corr_sum/=(HPconfs-1)
        MC_scores.append(corr_sum)

    return np.argmax(MC_scores)

def HITS(OD_scores,_):
    # score_mat: (n_samples, n_HPconfs)
    score_mat = np.stack(OD_scores,axis=-1)
    

    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat
    n_samples, n_HPconfs = score_mat.shape[0], score_mat.shape[1]

    hub_vec = np.full([n_HPconfs, 1],  1/n_HPconfs)
    auth_vec = np.zeros([n_samples, 1])

    hub_vec_list = []
    auth_vec_list = []

    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)

    for i in range(500):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)

        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)

        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]


        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            print('break at', i)
            break

        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)

    return np.argmax(hub_vec)



def all_model_select_algorithms(od_scores,AUCs,y):

    baseline_fn =[XB, MC,HITS]


    baseline_name = ['XB','MC','Hit']
    data= dict()
    for i in range(len(baseline_fn)):
        fn = baseline_fn[i]

        name = baseline_name[i]
        t0 = time()
        select_index = fn(od_scores,y)
        t1 = time()
        data[name] = AUCs[select_index]
        print('finish ',name,' in ',round(t1-t0,ndigits=4))


    print('\n')

    return data


def getY(data_name,type):
    if type =='tab':
        return np.load('../data/'+data_name+'.npz')['y']
    elif type =='img':
        return np.load('../data/'+data_name+'.npz')['y']

    

    