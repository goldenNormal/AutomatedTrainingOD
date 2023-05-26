import numpy as np
import sys
sys.path.append("..")
import os
from dataset_generator import generate_data,generate_numpy_data
from test_tube import HyperOptArgumentParser
import torch
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
#general settings
parser = HyperOptArgumentParser(strategy='grid_search')
parser.add_argument('--data', default='MNIST', help='currently support MNIST only')
parser.add_argument('--transductive', type = str2bool, default= True)
args=parser.parse_args()


norm_classes = [3,5]

from utils import set_seed
set_seed(0)

for norm_class in norm_classes:
    train_X, train_y =  generate_data(norm_class, dataset= args.data, transductive = True, flatten = True, GCN = True)

    np.savez_compressed('./data-for-primary/{}_{}.npz'.format(args.data,norm_class), X=train_X, y=train_y)