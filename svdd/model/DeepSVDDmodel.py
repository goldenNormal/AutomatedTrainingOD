import time

import torch
from sklearn.metrics import  roc_auc_score
import numpy as np

from tqdm import tqdm
from model.lenet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, MNIST_LeNet, MNIST_LeNet_Autoencoder
import os

from model.utils_for_svdd import cal_entropy

import torch.optim as optim

import numpy as np
import math
import torch

class CustomizeDataLoader():
    def __init__(self, 
                 data,
                 label = None,
                 num_models = 1,
                 batch_size = 300,
                 subsampled_indices = [],
                 device = "cuda"):
        super(CustomizeDataLoader, self).__init__()
        self.data = data
        self.label = label
        self.num_models = num_models
        self.total_batch_size = batch_size * num_models
        self.original_batch_size = batch_size
        self.device = device
            
        #we donot use subsampling here
        if subsampled_indices == []:
            self.subsampled_indices = [np.random.permutation(np.arange(self.data.shape[0]))] * num_models
        else:
            self.subsampled_indices = subsampled_indices
    
    def num_total_batches(self):
        return math.ceil(self.subsampled_indices[0].shape[0] /(self.original_batch_size) )
    
    def get_next_batch(self, idx):
        """
        Describe: Generate batch X and batch y according to the subsampled indices
        Parameter: 
             idx: the index of iteration in the current batch
        Return:
             batch_index: the indices of subsampling
             batch_X: numpy array with subsampled indices
        """    
        num_per_network = self.original_batch_size
        batch_index = [self.subsampled_indices[i][idx *num_per_network : (idx + 1)*num_per_network] 
                                                                           for i in range(self.num_models)]
        batch_index = np.concatenate(batch_index, axis=0)
        if self.label is not None:
            return batch_index, \
                   torch.tensor(self.data[batch_index]).to(self.device), \
                   self.label[batch_index]
        else:  
           # print(torch.from_numpy(self.data[batch_index]).shape)
            return batch_index, torch.tensor(self.data[batch_index]).to(self.device)

class ConvDeepSVDD():
    def __init__(self,
                 conv_dim=8,
                 fc_dim=32,
                 relu_slope=0.1,
                 pre_train=False,
                 pre_train_weight_decay=1e-6,
                 train_weight_decay=1e-6,
                 pre_train_epochs=100,
                 pre_train_lr=1e-4,
                 pre_train_milestones=[0],
                 train_epochs=250,
                 train_lr=1e-4,
                 train_milestones=[0],
                 batch_size=250,
                 device="cuda",
                 objective='one-class',
                 nu=0.1,
                 warm_up_num_epochs=10,
                 dataset="mnist"):

        if dataset == "mnist":
            self.ae_net = MNIST_LeNet_Autoencoder(conv_input_dim_list=[1, conv_dim, 4], fc_dim=fc_dim,
                                                  relu_slope=relu_slope)
        elif dataset == "cifar10":
            self.ae_net = CIFAR10_LeNet_Autoencoder(conv_input_dim_list=[1, conv_dim,  conv_dim,  conv_dim],
                                                    fc_dim= fc_dim, relu_slope=relu_slope)
        if dataset == "mnist":
            self.net = MNIST_LeNet(conv_input_dim_list=[1, conv_dim, 4], fc_dim=fc_dim, relu_slope=relu_slope)
        elif dataset == "cifar10":
            self.net = CIFAR10_LeNet()
        self.pre_train = pre_train
        self.device = device

        R = 0.0  # hypersphere radius R
        c = None  # hypersphere center c

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=device) if c is not None else None

        # Deep SVDD Hyperparameters
        self.nu = nu
        self.objective = objective
        self.batch_size = batch_size
        self.pre_train_weight_decay = pre_train_weight_decay
        self.train_weight_decay = train_weight_decay
        self.pre_train_epochs = pre_train_epochs
        self.pre_train_lr = pre_train_lr
        self.pre_train_milestones = pre_train_milestones
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.train_milestones = train_milestones
        self.warm_up_n_epochs = warm_up_num_epochs

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        num_total_batches = train_loader.num_total_batches()
        net.eval()
        with torch.no_grad():
            for idx in range(num_total_batches):
                # get the inputs of the batch
                _,inputs  = train_loader.get_next_batch(idx)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    
    def get_radius(self ,dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)
    
    def nai_train(self, train_data,y,dataset):
        if dataset == "mnist":
            train_data = train_data.reshape(-1,1,28,28)
        else:
            train_data = train_data.reshape(-1,3,32,32)
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_data,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)

        #pretrain the autoencoder
        # self.pre_train = False
        
        # Initilalize the net
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay, 
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(dataloader, self.net)
            print('Center c initialized.')

        self.net.train()

      
        num_total_batches = dataloader.num_total_batches()
        
    

        start = time.time()

        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0
            for idx in range(num_total_batches):
                _, inputs  = dataloader.get_next_batch(idx)
                start_time = time.time()
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)
        end = time.time()
        train_time = end - start


        outlier_score = self.predict(train_data)

        return roc_auc_score(y,outlier_score),train_time,outlier_score
    
    def opt_train(self, train_data,y,dataset):
        if dataset == "mnist":
            train_data = train_data.reshape(-1,1,28,28)
        else:
            train_data = train_data.reshape(-1,3,32,32)
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_data,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)

 
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay, 
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(dataloader, self.net)
            print('Center c initialized.')

        self.net.train()

      
        num_total_batches = dataloader.num_total_batches()
        
    

        start = time.time()

        AUC = []

        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0
            for idx in range(num_total_batches):
                _, inputs  = dataloader.get_next_batch(idx)
                start_time = time.time()
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)

                outlier_score = self.predict(train_data)
                auc = roc_auc_score(y,outlier_score)
                AUC.append(auc)
        end = time.time()
        train_time = end - start


       

        return np.max(AUC),train_time,None

    def en_train(self, train_data,y,dataset):
        if dataset == "mnist":
            train_data = train_data.reshape(-1,1,28,28)
        else:
            train_data = train_data.reshape(-1,3,32,32)
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_data,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)

        # Initilalize the net
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay, 
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(dataloader, self.net)
            print('Center c initialized.')

        self.net.train()

      
        num_total_batches = dataloader.num_total_batches()

        start = time.time()


        N_eval = min(1024, train_data.shape[0])
        eval_index = np.random.choice(train_data.shape[0], N_eval, replace=False)
        x_eval = train_data[eval_index]
        from EntropyEarlyStop import ModelEntropyEarlyStop
        ES = ModelEntropyEarlyStop(k=100,R_down=0.1)

        start = time.time()
        isStop = False
        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0
            for idx in range(num_total_batches):
                _, inputs  = dataloader.get_next_batch(idx)
                start_time = time.time()
                optimizer.zero_grad()
                self.net.train()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)


                isStop = ES.step(cal_entropy(self.predict(x_eval)),self.net)
                if isStop:
                    break
            if isStop:
                    break

        end = time.time()
        train_time = end - start

        self.net = ES.getBestModel()
        outlier_score = self.predict(train_data)

        return roc_auc_score(y,outlier_score),train_time, None
    


    def fastPredict(self, test_tensor):
        self.net = self.net.to(self.device)
        # Testing
        self.net.eval()
        inputs = test_tensor
        with torch.no_grad():
            outputs = self.net(inputs)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
            else:
                scores = dist

        return scores.detach().cpu().numpy()


    
    def predict(self, test_data):
        #load the overall data into a dataloader, so we can compute the score all together
        self.net = self.net.to(self.device)
        # Testing
        self.net.eval()
        inputs = torch.FloatTensor(test_data).to('cuda')
        with torch.no_grad():

            outputs = self.net(inputs)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
            else:
                scores = dist

        return scores.detach().cpu().numpy()
    
