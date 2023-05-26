import numpy as np
import torch
import random
def cal_entropy(score):
    score = score.reshape(-1)
    score = score/np.sum(score) # to possibility
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy


def dataLoading(path):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    # 归一化
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    print(x.shape)
    y = data['y']
    return x, y

def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_rdp_hps():
    out_cs =  [25, 50,100]
    lrs = [0.1, 0.01]
    dropouts = [0.0, 0.1]
    filter_ratios = [0.0, 0.05, 0.1]
    epochs = [100, 500]
    return out_cs, lrs, dropouts, filter_ratios, epochs