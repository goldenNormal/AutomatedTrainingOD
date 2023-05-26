import random
import torch
import numpy as np


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def dataLoading(path):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    y = data['y']
    print(x.shape)

    return x, y

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

def cal_entropy(score):
    score = score.reshape(-1)
    score = score/np.sum(score) # to possibility
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy