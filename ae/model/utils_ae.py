import torch
import numpy as np

def cal_entropy(score):
    score = score.reshape(-1)
    score = score/np.sum(score) # to possibility
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy

import random
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
    # standardization
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    print(x.shape)
    y = data['y']
    return x, y


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)


def get_ae_hps():
    act_funcs = ['relu', 'sigmoid']
    dropouts = [0, 0.2]
    h_dims = [64,256]
    num_layers = [1, 2]  # This is just the layer of the encoder
    lrs = [0.005, 0.001]
    epochs = [100, 500]

    return act_funcs, dropouts, h_dims, num_layers, lrs, epochs

def get_default_hp():
    n_eval = 1024
    
    batch_size = 256

    act_func = 'relu'
    dropout = 0.2
    h_dim = 64
    num_layer = 1  # This is just the layer of the encoder
    lr = 0.001
    epoch = 250
    k = 100
    r_down = 0.1


    return n_eval, batch_size, act_func, dropout, h_dim, num_layer, lr, epoch, k, r_down