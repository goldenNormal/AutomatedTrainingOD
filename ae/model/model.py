import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

    
class ParameterAE(nn.Module):
    def __init__(self,
                 in_dim,h_dim=64,num_layer=1,dropuout=0.2,first_layer_act='relu'):
        '''ae HPs, including h_dim, dropout, first_layer_act(relu or sigmoid), num_layer'''
        '''num_layer refers to the number of layers of the encoder, and the number of layers of the decoder is the mirror image of the encoder'''
        super(ParameterAE, self).__init__()
        self.dropout = nn.Dropout(p = dropuout)
        
        #input output dim initialization
        self.input_dim_list = [in_dim] + [h_dim for i in range(num_layer)] 
        self.output_dim_list  = self.input_dim_list[::-1][1:]
        self.dim_list = self.input_dim_list + self.output_dim_list
        self.n_layer = len(self.dim_list)

        self.layer_list = nn.ModuleList() 
        for i in range(self.n_layer - 1):
            self.layer_list.append(nn.Linear(in_features= self.dim_list[i],
                                                      out_features = self.dim_list[i+1]))
        self.first_layer_act = first_layer_act

              
    def forward(self, x):
        ori_x = x
        x = self.layer_list[0](x)
        if self.first_layer_act == 'relu':
            
            x = torch.relu(x)
        elif self.first_layer_act == 'sigmoid':
            x = torch.sigmoid(x)
        else:
            raise ValueError('first_layer_act should be relu or sigmoid')

        x = self.dropout(x)
        
        for i in range(1, self.n_layer -2):
            x = self.layer_list[i](x)
            x = self.dropout(torch.relu(x))
        
        output = self.layer_list[-1](x)

        return torch.sum(torch.square(output - ori_x),dim=-1)
    
