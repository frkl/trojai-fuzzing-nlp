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

import torch.optim as optim

from transformers import RobertaTokenizer, RobertaForMaskedLM

class Fuzzer():
    def __init__(self):
        while True:
            try:
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base");
                break;
            except:
                pass;
        
        self.pad=self.encode(self.tokenizer.pad_token)[0];
        self.vocab_size=len(self.tokenizer);
        self.special_tokens=set([self.encode(x)[0] for x in [self.tokenizer.bos_token,self.tokenizer.eos_token,self.tokenizer.unk_token,self.tokenizer.sep_token,self.tokenizer.pad_token,self.tokenizer.cls_token,self.tokenizer.mask_token,]+self.tokenizer.additional_special_tokens]);
        self.valid_tokens=list(set(range(self.vocab_size)).difference(self.special_tokens));
        
        if not surrogate is None:
            self.surrogate=surrogate;
        
    
    def encode(self,data):
        if isinstance(data,list):
            return [self.tokenizer.encode(s)[1:-1] for s in data];
        elif isinstance(data,str):
            return self.tokenizer.encode(data)[1:-1];
        else:
            print('unrecognized input');
            a=0/0;
    
    def decode(self,data):
        if isinstance(data,list):
            if len(data)==0:
                return self.tokenizer.decode(data);
            elif isinstance(data[0],list):
                return [self.tokenizer.decode([t for t in s if not t==self.pad]) for s in data];
            else:
                return self.tokenizer.decode([t for t in data if not t==self.pad]);
        elif torch.is_tensor(data):
            return self.decode(data.tolist());
        else:
            print('unrecognized input');
            a=0/0;
    
    def generate_random_sequence(self,length=1,decode=False,n=-1,maxl=None):
        valid_tokens=torch.LongTensor(self.valid_tokens);
        if maxl is None:
            maxl=length;
        
        if n>=1:
            x=torch.LongTensor(n,length).random_(len(self.valid_tokens));
        else:
            x=torch.LongTensor(length).random_(len(self.valid_tokens));
        
        tokens=valid_tokens[x];
        tokens=F.pad(tokens,(0,maxl-length),value=self.pad);
        
        if decode:
            tokens=self.decode(tokens)
        return tokens;
    
    def create_surrogate(self,*args,**argv):
        #Load initial word embeddings
        while True:
            try:
                roberta=RobertaForMaskedLM.from_pretrained('roberta-base');
                break;
            except:
                pass;
        
        we=roberta.roberta.embeddings.word_embeddings.weight;
        return surrogate(we,*args,**argv);
     
    def suggest(self,x,y,l=8,target=None):
        #Assuming surrogate has registered we
        x=x.to(self.surrogate.reg.device)
        x=x.to(self.surrogate.reg.device)
        y=y.to(self.surrogate.reg.device)
        
        nobs=y.shape[-1];
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best=y.mean(dim=1).max();
        #print(current_best)
        
        #Tool for checking what options are available for tokens[ind]
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
        
        
        with torch.no_grad():
            tokens=torch.LongTensor(self.surrogate.maxl).fill_(self.pad);
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
                        qy_std_avg=qy_std.mean(dim=-1); #Looks incorrect but actually correct...
                        
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

class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0))
    
    # X: B N K
    # Y: B N Y
    def learn(self,X,Y):
        reg=torch.exp(self.reg);
        N=Y.shape[-2];
        mu=Y.mean(-2,keepdim=True);
        s=Y.std(-2,keepdim=True)+1e-3;
        l2=-torch.cdist(X.transpose(-2,-1),((Y-mu)/s).transpose(-2,-1)); # B K Y
        p=F.softmax(l2,dim=-2);
        return p,mu,s;
    
    def predict(self,X,p,mu,s):
        k=torch.matmul(X,p)
        y=k*s+mu;
        kstd=torch.matmul(X**2,p)
        ystd=(kstd-k*k).clamp(min=0)**0.5*s
        return y,ystd;


class surrogate(nn.Module):
    def __init__(self,we,maxl=8,std=False):
        super(surrogate,self).__init__()
        we=we.data.clone().float();
        nhword=we.shape[1];
        
        nh=512;
        nlayers=3;
        
        self.we=nn.Parameter(we);
        self.encoder=MLP(nhword*maxl,512,nh,nlayers);
        self.regressor=regressor();
        
        self.nhword=nhword;
        self.nh=nh;
        self.nlayers=nlayers;
        self.maxl=maxl;
        self.std=std;
        
        return;
    
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
    
    def forward(self,x,y,qx):
        e=self.embed(x); #Nxnh
        ws=self.regressor.learn(e,y);
        
        e=self.embed(qx);
        qy,qy_std=self.regressor.predict(e,*ws);
        return qy;
