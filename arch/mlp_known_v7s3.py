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


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=int(params.nh**0.5)*4;
        self.q0=torch.arange(0,1+1e-20,1/nh).cuda()
        nh=len(self.q0);
        
        nh2=int(params.nh2**0.5)
        self.q1=torch.arange(0,1+1e-20,1/nh2).cuda()
        nh2=len(self.q1);
        
        nh3=params.nh3;
        
        if params.nlayers>1:
            self.encoder1=MLP(nh,nh,nh,params.nlayers-1);
            self.encoder2=MLP(nh*nh2,nh3,2,params.nlayers2);
        else:
            self.encoder1=nn.Identity();
            self.encoder2=MLP(nh*nh2,nh3,2,params.nlayers2);
        
        
        self.encoder1=nn.Identity();
        self.encoder2=MLP(nh*nh2,nh3,2,params.nlayers2);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch):
        x=[v.cuda()[:-1,:].contiguous().t() for v in data_batch['score']]
        x=[torch.quantile(v,self.q0,dim=1) for v in x];
        x=[self.encoder1(v) for v in x];
        x=[torch.quantile(v,self.q1,dim=1) for v in x];
        x=[v.view(-1) for v in x];
        h=torch.stack(x,dim=0);
        h=self.encoder2(h)
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];