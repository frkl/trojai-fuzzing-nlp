import torch
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy


class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        h=h.view(*(list(x.shape[:-1])+[-1]));
        return h


class pool_encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput+2,nh,nh,nlayers);
        self.ninput=ninput;
        self.nh=nh;
        self.nlayers=nlayers;
        return;
    
    def forward(self,x):
        #x: N x K matrix
        #Sort K, encode
        #=> N x nh
        h=x.sort(dim=-1)[0];
        h=torch.cat((h[:,0:1],h[:,-1:],F.adaptive_avg_pool1d(h.unsqueeze(0),self.ninput).squeeze(0)),dim=-1)
        h=self.encoder(h)
        return h


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2
        nh3=params.nh3
        
        #self.budget=params.nh3//4;
        
        self.encoder1=pool_encoder(nh//4,nh2,params.nlayers);
        #self.encoder2=pool_encoder(nh2,nh2,params.nlayers2);
        self.encoder3=MLP(3*nh2,nh3,2,params.nlayers3)
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch): # triggers x nobs
        x=[v.cuda().t() for v in data_batch['score']] 
        x=[(v-v.mean(dim=1,keepdim=True))/(v.std(dim=1,keepdim=True)+torch.exp(self.reg*10)) for v in x]
        #x=[(v-v.median(dim=1,keepdim=True)[0])/(v.max(dim=1,keepdim=True)[0]-v.min(dim=1,keepdim=True)[0]+torch.exp(self.reg*10)) for v in x]
        #x=[F.normalize(v-v[:,0:1],dim=1) for v in x]
        x=[self.encoder1(v) for v in x]; # nobs x nhtrig
        #x=[self.encoder2(v.t()) for v in x]; # nhtrig x nobs
        x=[torch.cat((v.max(dim=0)[0],v.min(dim=0)[0],v.mean(dim=0)),dim=0) for v in x]
        #x=[v.mean(dim=0) for v in x];
        h=torch.stack(x,dim=0);
        h=self.encoder3(h)
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];