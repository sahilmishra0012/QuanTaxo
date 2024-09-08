import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class MLP_VEC(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP_VEC,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.nlnr = nn.ReLU()
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self,x,printit=False):
        x = self.fc1(x)
        x = self.nlnr(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
    
class NormalisedWeights(nn.Module):
    def __init__(self, input_dim, mixturetype=None):
        super(NormalisedWeights,self).__init__()
        self.type = mixturetype
        self.input_dim = input_dim
        if mixturetype == "random":
            self.weights = nn.Parameter(torch.randn(input_dim))
        elif mixturetype == "constant":
            self.weights = nn.Parameter(torch.ones(input_dim))
        elif mixturetype == "uniform":
            self.weights = nn.Parameter(torch.empty(input_dim).uniform_(-1,1))
        elif mixturetype == "self_attention":
            self.weights = nn.Parameter(torch.empty((input_dim//2,768)))
            self.a = nn.Parameter(torch.empty(input_dim))
        elif mixturetype == "nn":
            self.weights = nn.Parameter(torch.randn(input_dim))
            self.ann = nn.Linear(input_dim, input_dim, bias=True)
        elif mixturetype == "qen":
            self.weights = 0
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)
        self.fc1 = nn.Softmax(dim=-1)

    def forward(self,x):
        # Normalise the weights to sum to 1
        if self.type=="self_attention":
            weight_mat = self.weights.T
            print("Wt mat:",weight_mat.shape)
            a = self.a
            print("a param: ",a.shape)
            weighted_states = torch.matmul(x,weight_mat)
            print("Wt st:",weighted_states.shape)
            mat1 = weighted_states.unsqueeze(-2).repeat(1,1,self.input_dim,1)
            print("M1",mat1.shape)
            mat2 = weighted_states.unsqueeze(-3).repeat(1,self.input_dim,1,1)
            print("M2: ",mat2.shape)
            w_concat = torch.cat([mat1,mat2],dim=-1)
            print("W cat: ",w_concat.shape)
            w_concat = torch.matmul(w_concat,a)
            print("W cat2: ",w_concat.shape)
            w_concat = self.lrelu(w_concat)
            print("W cat3: ",w_concat.shape)
            weights = self.fc1(w_concat)
            print("Weight: ",weights.shape)
            weighted_sum_batch = torch.matmul(weights,weighted_states)
            print(weighted_sum_batch.shape)
        elif self.type=="nn":
            weights = self.ann(self.weights)
            weights = self.fc1(weights)
            weighted_sum_batch = torch.einsum('bij,i->bj', x, weights)
        else:
            weights = self.fc1(self.weights) # Get the weights sum to 1
            # print(weights,"\nand sum = ",weights.sum())
            assert(x.shape[1]==weights.shape[0])
            weighted_sum_batch = torch.einsum('bij,i->bj', x, weights)
            # print(x.shape)
            print(weighted_sum_batch.shape)
        return weighted_sum_batch


class LINEAR_ONE(nn.Module):
    def __init__(self,input_dim):
        super(LINEAR_ONE,self).__init__()
        self.fc1 = nn.Linear(input_dim,input_dim,bias=True)
        self.fc2 = nn.Softmax(dim=-1)
        # self.fc1 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self,x):
        y = self.fc1(x)
        y = self.fc2(y)
        x = torch.outer(x,x) # Construct the density matrix for each 
        x = F.linear(x,y)
        return x