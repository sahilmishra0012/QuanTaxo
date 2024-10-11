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
    
class PhaseEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(PhaseEmbeddings,self).__init__()
        self.weights = nn.Parameter(torch.empty(embed_dim,2).uniform_(-1,1))
        self.sftmx = nn.Softmax(dim=-1)
    
    def forward(self):
        phase = self.sftmx(self.weights)
        phase = torch.view_as_complex(phase)
        return phase
        
class NormalisedWeights(nn.Module):
    def __init__(self, input_dim, mixturetype=None, modeltype=None):
        super(NormalisedWeights,self).__init__()
        self.type = mixturetype
        self.valspace = "complex" if modeltype else "real"
        if mixturetype == "constant":
            self.weights = torch.ones(input_dim).cuda()
        elif mixturetype == "uniform":
            self.weights = nn.Parameter(torch.empty(input_dim).uniform_(-1,1))
        self.fc1 = nn.Softmax(dim=-1)

    def forward(self,x):
        # Normalise the weights to sum to 1
        weights = self.fc1(self.weights) 
        assert(x.shape[1]==weights.shape[0])
        if self.valspace=="complex":
            weights = torch.sqrt(weights)
        return weights