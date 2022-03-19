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


def kernel(x,x2):
    K=torch.matmul(x,x2.transpose(-1,-2));
    return K

def kernel_z(x):
    Kz=x*x;
    Kz=Kz.sum(dim=-1,keepdim=True);
    return Kz;



def ridge_learn(X,Y,reg):
    #Convert floats to double
    X=X.type(reg.data.dtype) #bnh
    Y=Y.type(reg.data.dtype) #bny
    
    N=X.shape[-2];
    nh=X.shape[-1];
    
    regI=reg*torch.eye(nh).to(reg.device);
    if len(X.shape)>=3:
        regI=regI.view([1]*(len(X.shape)-2)+[nh,nh]);
    
    X=X.double();
    Y=Y.double();
    b=Y.mean(-2,keepdim=True); #b1y
    A=torch.matmul(X.transpose(-1,-2),X) + regI.double() # bhh
    a=torch.matmul(X.transpose(-1,-2),Y-b); #bhy
    w=torch.matmul(torch.inverse(A.double()),a); #bhy
    
    w=w.type(reg.data.dtype);
    b=b.type(reg.data.dtype);
    return w,b

def ridge_predict(X,w,b):
    #Convert floats to double
    X=X.type(w.data.dtype) #bnh
    Y=torch.matmul(X,w)+b
    return Y,Y*0+1



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


class regressor(nn.Module):
    def __init__(self,N,nh):
        super().__init__()
        self.key=nn.Parameter(torch.Tensor(N,nh).uniform_(-1/math.sqrt(nh),1/math.sqrt(nh)));
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def forward(self,w):
        w,b=w;
        h=torch.matmul(self.key,w)#+b
        return h
    

class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2
        
        self.budget=1800;
        self.N=nh*2;
        nhinput=512;
        
        self.encoder0=regressor(self.N,nhinput);
        
        if params.nlayers>1:
            self.encoder1=MLP(self.N,nh,nh,params.nlayers-1);
            self.encoder2=MLP(2*nh,nh2,2,params.nlayers2);
        else:
            self.encoder1=nn.Identity();
            self.encoder2=MLP(2*self.N,nh2,2,params.nlayers2);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        self.margin=params.margin;
        return;
    
    def forward(self,data_batch):
        x=[self.encoder0(v) for v in data_batch['ws']];
        x=[self.encoder1(v.t()) for v in x];
        x=[torch.cat((v.max(dim=0)[0],v.min(dim=0)[0]),dim=0) for v in x]
        h=torch.stack(x,dim=0);
        h=self.encoder2(h)
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];