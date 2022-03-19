import torch
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy
#from performer_pytorch import Performer
from fast_transformers.builders import TransformerEncoderBuilder

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
    def __init__(self,nh,nlayers):
        super().__init__()
        if nlayers==1:
            self.encoder=nn.Identity();
        else:
            self.encoder=MLP(nh,nh,nh,nlayers-1);
        self.nh=nh;
        self.nlayers=nlayers;
        return;
    
    def forward(self,x):
        #x: N x K matrix
        #Sort K, encode
        #=> N x nh
        h=x.sort(dim=-1)[0];
        h=torch.cat((h[:,0:1],h[:,-1:],F.adaptive_avg_pool1d(h.unsqueeze(0),self.nh-2).squeeze(0)),dim=-1)
        h=self.encoder(h)
        return h


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=(params.nh//8)*4;
        nh2=params.nh2
        nh3=params.nh3
        self.nh=nh
        
        #self.budget=params.nh3//4;
        self.t0=MLP(1800,self.nh,self.nh,1);
        if params.nlayers>1:
            nlayers=min(params.nlayers-1,2);
            #layer=nn.TransformerEncoderLayer(nh,nhead=4,dim_feedforward=nh,dropout=0);
            #self.transformer = nn.TransformerEncoder(layer,params.nlayers-1);
            self.transformer = TransformerEncoderBuilder.from_kwargs(n_layers=nlayers,n_heads=1,query_dimensions=nh,value_dimensions=nh,feed_forward_dimensions=nh,attention_type="full",activation="gelu").get()
        else:
            self.transformer = nn.Identity()
        
        self.encoder=MLP(3*nh,nh2,2,params.nlayers2)
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def forward(self,data_batch): # triggers x nobs
        x=[self.t0(v.cuda().t().contiguous()) for v in data_batch['score']] 
        #x=[(v-v.mean(dim=0,keepdim=True))/(v.std(dim=0,keepdim=True)+1e-10) for v in x] #Normalize along triggers
        #x=[v.sort(dim=-1)[0] for v in x];
        #x=[torch.cat((v[:,0:1],F.adaptive_avg_pool1d(v.unsqueeze(0),self.nh-2).squeeze(0),v[:,-1:]),dim=-1) for v in x];
        x=[self.transformer(v.unsqueeze(0)).squeeze(0) for v in x]
        
        x=[torch.cat((v.max(dim=0)[0],v.min(dim=0)[0],v.mean(dim=0)),dim=0) for v in x]
        
        h=torch.stack(x,dim=0);
        h=self.encoder(h)
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];