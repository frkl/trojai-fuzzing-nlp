#bots_roberta_x1N: 
#Trigger search:
#    Roberta tokenizer for text generation
#    Uncertainty-based search
#    Query examples with large uncertainty std, average across output dimensions
#    Learned surrogate model for uncertainty evaluation
#Surrogate
#    3-layer MLP for input embedding
#    Linear GP surrogate
#    Mean/std-normalize input before regression

import torch
import torch.linalg
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy
import time
import os

import torch.optim as optim

from transformers import RobertaTokenizer, RobertaForMaskedLM

class new():
    #Initialize empty trigger search method
    def __init__(self,maxl=8):
        nhword=768 #Roberta embedding size
        
        #Load roberta tokenizer for word synthesis
        for n in range(5):
            try:
                self.tokenizer=torch.load('roberta_tokenizer.pt');
                break
            except:
                try:
                    self.tokenizer=torch.load('/roberta_tokenizer.pt');
                    break
                except:
                    try:
                        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base");
                        torch.save(self.tokenizer,'roberta_tokenizer.pt')
                        break
                    except:
                        print('Failed to load tokenizer for trigger search module, try again');
                        pass;
        
        self.pad=self.encode(self.tokenizer.pad_token)[0];
        self.vocab_size=len(self.tokenizer);
        self.special_tokens=set([self.encode(x)[0] for x in [self.tokenizer.bos_token,self.tokenizer.eos_token,self.tokenizer.unk_token,self.tokenizer.sep_token,self.tokenizer.pad_token,self.tokenizer.cls_token,self.tokenizer.mask_token,]+self.tokenizer.additional_special_tokens]);
        self.valid_tokens=list(set(range(self.vocab_size)).difference(self.special_tokens));
        
        self.maxl=maxl;
        self.surrogate=surrogate(nhword=nhword,N=self.vocab_size,maxl=maxl).cuda();
    
    
    #Tokenize strings or list of strings to ids
    def encode(self,data):
        if isinstance(data,list):
            return [self.tokenizer.encode(s)[1:-1] for s in data];
        elif isinstance(data,str):
            return self.tokenizer.encode(data)[1:-1];
        else:
            print('encode: unrecognized input');
            a=0/0;
    
    #Decode ids into strings or list of strings
    def decode(self,data):
        if isinstance(data,list):
            if len(data)==0: #One empty sentence
                return self.tokenizer.decode(data);
            elif isinstance(data[0],list): #list of sentences
                return [self.tokenizer.decode([t for t in s if not t==self.pad]) for s in data];
            else: #One single sentence
                return self.tokenizer.decode([t for t in data if not t==self.pad]);
        elif torch.is_tensor(data):
            return self.decode(data.tolist());
        else:
            print('decode: unrecognized input');
            a=0/0;
    
    def generate_random_sequence(self,length=1,decode=False,n=-1,maxl=None):
        valid_tokens=torch.LongTensor(self.valid_tokens);
        if maxl is None:
            maxl=length;
        
        if n>=1: #Need batch of sequences
            x=torch.LongTensor(n,length).random_(len(self.valid_tokens));
        else: #Need single sequence
            x=torch.LongTensor(length).random_(len(self.valid_tokens));
        
        tokens=valid_tokens[x];
        tokens=F.pad(tokens,(0,maxl-length),value=self.pad); #Pad to same length for easy tensor conversion
        
        if decode: #Return text if requested
            tokens=self.decode(tokens)
        return tokens;
    
    #Input
    #    X: N triggers x L ids
    #    Y: N triggers x K scores
    #    l: expected trigger length
    #Returns: 
    #    list of suggested triggers
    def suggest(self,x,y,l=None):
        if l is None:
            l=self.maxl;
        
        if l>self.maxl:
            l=self.maxl;
        
        #Generate random sequences in the first iteration
        if (x is None or len(x)==0) and (y is None or len(y)==0):
            queries=[];
            for i in range(1,l+1):
                queries.append(self.generate_random_sequence(length=i,maxl=self.maxl).tolist())
            
            return queries,[0 for q in queries];
        
        #Assuming surrogate has registered we
        if isinstance(x,list):
            x=torch.LongTensor(x);
        
        if isinstance(y,list):
            y=torch.Tensor(y);
        
        x=x.to(self.surrogate.device())
        y=y.to(self.surrogate.device())
        ws=self.surrogate.regress(x,y);
        
        #Utility for avoiding duplicates
        #    Checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        valid_tokens=set(self.valid_tokens)
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=valid_tokens;
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=valid_tokens.difference(set(dupe_tokens));
            
            available_tokens=list(available_tokens);
            return available_tokens;
        
        #Generate 1 candidate for each length
        best_scores=[];
        with torch.no_grad():
            tokens=torch.LongTensor(self.maxl).fill_(self.pad);
            candidates=[];
            for i in range(1,l+1):
                best_score=1e10;
                tokens[0:i]=torch.LongTensor(self.generate_random_sequence(i));
                while True:
                    improved=False
                    for ind in range(0,i):
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        qy_avg=qy.mean(dim=-1);
                        qy_std_avg=torch.log(qy_std.clamp(min=1e-4)).mean(dim=-1); #Looks incorrect but actually correct...
                        
                        #Find largest uncertainty
                        s=-qy_std_avg;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores


#Load a checkpoint
def load(checkpoint):
    m=new();
    try:
        checkpoint=torch.load(checkpoint);
    except:
        checkpoint=torch.load(os.path.join('/',checkpoint));
    
    m.surrogate.load_state_dict(checkpoint);
    m.surrogate=m.surrogate.float();
    m.surrogate.register();
    m.surrogate.eval()
    return m;




#Basic MLP
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
        
        self.pre=False;
        self.embeddings=[];
        
        return;
    
    def forward(self,x):
        if isinstance(x,list):
            #Use precalculated embedding lookup
            e=[];
            for i in range(len(x)):
                if isinstance(x[i],int):
                    e_i=self.embedding[i][x[i]:x[i]+1,:];
                else:
                    e_i=self.embedding[i][x[i].view(-1),:];
                e.append(e_i);
            
            #Add up the embeddings
            e_=e[0];
            for i in range(1,len(x)):
                e_=e_+e[i];
            
            e_=e_+self.layers[0].bias.data.view(1,-1);
            h=e_;
            if len(self.layers)>=2:
                h=F.relu(h);
                for i in range(1,len(self.layers)-1):
                    h=self.layers[i](h);
                    h=F.relu(h);
                    #h=F.dropout(h,training=self.training);
                
                h=self.layers[-1](h);
            
            return h
        
        else:
            h=x.view(-1,self.ninput);
            #h=self.bn(h);
            for i in range(len(self.layers)-1):
                h=self.layers[i](h);
                h=F.relu(h);
                #h=F.dropout(h,training=self.training);
            
            h=self.layers[-1](h);
            h=h.view(*(list(x.shape[:-1])+[-1]));
        
        return h
    
    def pre_multiply(self,we):
        nh=we.shape[1];
        
        #Check how many words are there in the input
        n=self.layers[0].weight.shape[1]//nh;
        
        #Convert layer 0 into embeddings
        self.pre=True;
        self.embedding=[];
        for i in range(n):
            e=torch.mm(we,self.layers[0].weight.data[:,i*nh:(i+1)*nh].t());
            self.embedding.append(e.data);
        
        return;


#Fully featured mixture GP
#No normalization cus it's bad?
#    Could add in e3N

#T: temperature on L2 loss
#reg: per-component per-dimension regularization
#w0: per component initial weight
#bias: per component logp bias, to make sure that all components will be used 


#Let's start with a batch 1 version
def ridge_learn(X,Y,reg,t,bias,wz,training=False):
    #Convert floats to double
    X=X.type(reg.data.dtype) #bnh
    Y=Y.type(reg.data.dtype) #bny
    
    #X: bnh
    #Y: bny
    shape=X.shape
    N=X.shape[-2];
    G=len(bias);
    nh=X.shape[-1]//G;
    ny=Y.shape[-1];
    
    X=X.view(-1,N,G*nh);
    Y=Y.view(-1,N,ny);
    
    #Adjust Y by wz
    GX=X.view(-1,N,G,nh).transpose(-2,-3).contiguous().view(-1,N,nh); #BG N nh
    GY=(Y.view(-1,1,N,ny).repeat(1,G,1,1)-(GX.view(-1,G,N,nh)*wz.view(1,G,1,nh)).sum(dim=-1,keepdim=True)).view(-1,N,ny); #BG N ny #
    GX=GX.double();
    GY=GY.double();
    t=t.double();
    bias=bias.double();
    wz=wz.double()
    
    #Perform regression
    I=torch.diag_embed(reg).view(1,G,nh,nh);
    
    cov_inv=t*torch.matmul(GX.transpose(-2,-1).double(),GX.double()).view(-1,G,nh,nh)+t*I.double(); #BG nh nh
    cov_inv=cov_inv.view(-1,nh,nh);
    cov=torch.inverse(cov_inv) #BG nh nh
    a=torch.matmul(GX.transpose(-1,-2),GY); #BG nh ny
    w0=t*torch.matmul(cov,a); #BG nh ny
    
    #Calculate mixture weights
    S=torch.linalg.slogdet(cov_inv).logabsdet; #BG
    #Residual method 1
    residual=t*(GY*GY).sum(dim=-2).view(-1,G,ny) - (w0*torch.matmul(cov,w0)).sum(dim=-2).view(-1,G,ny); # B G Y
    logp=-0.5*(residual+S.view(-1,G,1));
    logp=F.log_softmax(logp-bias.view(1,G,1),dim=-2); #B G Y
    p=F.softmax(logp,dim=-2); #B G Y
    
    
    
    w0=w0.type(reg.data.dtype);
    cov=cov.type(reg.data.dtype);
    logp=logp.type(reg.data.dtype);
    p=p.type(reg.data.dtype);
    return w0,cov,logp,p

def ridge_predict(X,w0,cov,logp,p,training=False):
    X=X.type(w0.data.dtype)
    
    shape=X.shape;
    G=p.shape[1];
    N=X.shape[-2];
    nh=X.shape[-1]//G;
    ny=p.shape[-1]
    
    GX=X.view(-1,N,G,nh).transpose(-2,-3).contiguous().view(-1,N,nh); #BG N nh
    GY=torch.matmul(GX,w0) #BG N ny
    
    Y=(GY.view(-1,G,N,ny)*p.view(-1,G,1,ny)).sum(dim=-3) # B N ny
    dY0=GY.view(-1,G,N,ny)-Y.view(-1,1,N,ny);
    Y0var=((dY0**2)*p.view(-1,G,1,ny)).sum(dim=-3).view(-1,N,ny).clamp(min=1e-6);
    
    
    Yvar=(torch.matmul(GX,cov)*GX).sum(dim=-1).view(-1,G,N) #(BG N,1) #var of y within each Gaussian mode
    Yvar=(Yvar.view(-1,G,N,1)*p.view(-1,G,1,ny)).sum(-3).clamp(min=1e-6) #B G ny
    
    Yvar=Yvar+Y0var
    Ystd=Yvar**0.5;
    
    
    Y=Y.view(*shape[:-2],N,ny);
    Ystd=Ystd.view(*shape[:-2],N,ny);
    
    return Y,Ystd


def ridge_p(X,Ygt,w0,cov,logp,p,training=False):
    X=X.type(w0.data.dtype)
    Ygt=Ygt.type(w0.data.dtype)
    
    shape=X.shape;
    G=logp.shape[1];
    N=X.shape[-2];
    nh=X.shape[-1]//G;
    ny=logp.shape[-1]
    
    GX=X.view(-1,N,G,nh).transpose(-2,-3).contiguous().view(-1,N,nh); #BG N nh
    GY=torch.matmul(GX,w0) #BG N ny
    
    Y=(GY.view(-1,G,N,ny)*p.view(-1,G,1,ny)).sum(dim=-3) # B N ny
    Yvar=(torch.matmul(GX,cov)*GX).sum(dim=-1).view(-1,G,N) #(BG N,1) #var of y within each Gaussian mode
    Yvar=(Yvar.view(-1,G,N,1)*p.view(-1,G,1,ny)).sum(-3).clamp(min=1e-0) #B G ny
    
    z=(Ygt.view(-1,1,N,ny)-GY.view(-1,G,N,ny))/(Yvar.view(-1,G,N,1)**0.5);
    S=torch.log(Yvar.view(-1,G,N,1));
    logpgt=-0.5*S-0.5*z**2-0.9189385332;
    logpgt=logpgt+logp.view(-1,G,1,ny);
    logpgt=torch.logsumexp(logpgt,dim=1); #B N ny
    
    Y=Y.view(*shape[:-2],N,ny);
    logpgt=logpgt.view(*shape[:-2],N,ny);
    
    return Y,logpgt

class surrogate(nn.Module):
    def __init__(self,we=None,maxl=8,N=None,nh=1024,G=8,nlayers=3,nhword=None,pretrained=False):
        super(surrogate,self).__init__()
        
        #Load initial word embeddings
        if pretrained:
            roberta=RobertaForMaskedLM.from_pretrained('roberta-base');
            we=roberta.roberta.embeddings.word_embeddings.weight.data.float().clone();
        elif we is None:
            we=torch.Tensor(N,nhword).uniform_(-1/nh**0.5,1/nh**0.5);
        
        nhword=we.shape[1];
        self.we=nn.Parameter(we);
        
        self.encoder=MLP(nhword*maxl,512,nh,nlayers);
        
        
        self.t=nn.Parameter(torch.Tensor(1).fill_(0));
        self.bias=nn.Parameter(torch.Tensor(G).fill_(0));
        self.w0=nn.Parameter(torch.Tensor(G,nh//G).fill_(0));
        self.reg=nn.Parameter(torch.Tensor(G,nh//G).fill_(0));
        self.s=nn.Parameter(torch.Tensor(1).fill_(0));
        
        
        self.nhword=nhword;
        self.nh=nh;
        self.nlayers=nlayers;
        self.maxl=maxl;
        
        return;
    
    def device(self):
        return self.reg.device;
    
    def register(self,*args,**kwargs):
        self.encoder.pre_multiply(self.we);
    
    def embed(self,x):
        if isinstance(x,list):
            return self.encoder(x);
        else:
            e=self.we[x.view(-1),:];
            e=e.view(*(list(x.shape)[:-1]+[-1]));
            e=self.encoder(e)
            return e;
        
        return e;
    
    #Kernel GP regression
    #x: n x k, y: n x 1
    #Output embedding e for test kernel, a for kernel weight, b for classifier bias, A for uncertainty calcs
    def regress(self,x,y):
        reg=torch.exp((self.reg*10).clamp(min=-20,max=20))+1e-8;
        t=torch.exp((self.t*10).clamp(min=-20,max=20))+1e-8;
        s=torch.exp((self.s*10).clamp(min=-20,max=20))+1e-8;
        e=self.embed(x); #Nxnh
        if self.training:
            w0=ridge_learn(e,y,reg,t,self.bias*10,self.w0*10,training=self.training);
        else:
            #only tune temperature t
            w0=ridge_learn(e.data,y,reg,t,self.bias*10,self.w0*10,training=self.training);
        
        return (w0,);
    
    
    def score(self,qx,w0):
        e=self.embed(qx);
        if self.training:
            qy,qy_std=ridge_predict(e,*w0,training=self.training); #ny
        else:
            qy,qy_std=ridge_predict(e.data,*w0,training=self.training); #ny
        
        return qy,qy_std;
    
    def forward(self,x,y,qx):
        ws=self.regress(x,y);
        qy,qy_std=self.score(qx,*ws);
        #qy=qy.type(y.dtype);
        #qy_std=qy_std.type(y.dtype);
        return qy,qy_std;
    
    
    def score_logp(self,qx,qy,w0):
        e=self.embed(qx);
        qy_pred,logp=ridge_p(e,qy,*w0,training=self.training); #ny
        return qy_pred,logp;
    
    def forward_logp(self,x,y,qx,qy):
        ws=self.regress(x,y);
        qy_pred,qy_logp=self.score_logp(qx,qy,*ws);
        return qy_pred,qy_logp;

    