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
        nh=params.nh;
        nh2=params.nh2
        
        self.budget=params.nh3;
        
        self.encoder1=nn.Identity();
        self.encoder2=MLP(3*(self.budget+2),nh2,2,params.nlayers2);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch):
        x=[v.cuda().t() for v in data_batch['score']]
        x=[(v-v.median(dim=1,keepdim=True)[0]).abs() for v in x]
        x=[v.sort(dim=1)[0] for v in x];
        x=[torch.cat((v[:,0:1],v[:,-1:],F.adaptive_avg_pool1d(v.unsqueeze(0),self.budget).squeeze(0)),dim=-1) for v in x];
        x=[self.encoder1(v) for v in x];
        x=[torch.cat((v.max(dim=0)[0],v.min(dim=0)[0],v.mean(dim=0)),dim=0) for v in x]
        h=torch.stack(x,dim=0);
        h=self.encoder2(h)
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];