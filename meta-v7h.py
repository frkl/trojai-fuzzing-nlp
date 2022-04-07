
#import torch
#from performer_pytorch import Performer

#model=Performer(dim=256,dim_head=256,depth=12,heads=8).cuda();

#x=torch.Tensor(1,2048,256).requires_grad_().cuda(); #
#y=model(x);
#
#y.mean().backward()

#print(y.shape)


#Learn a token_gen using meta-learning
#Such that obj decrease is accelerated on given data

#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import math
import time
import random
import argparse
import sys
import os
import re
import copy
import importlib
import json
from collections import namedtuple
from collections import OrderedDict
from itertools import chain
import util.db as db

import os
import numpy as np
import copy
import torch
import transformers

import warnings
warnings.filterwarnings("ignore")
import util.db as db

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import random
from functools import partial

import util.session_manager as session_manager
import util.smartparse as smartparse
import util.file
import pandas


#torch.set_default_dtype(torch.float64)

# Training settings
def default_params():
    params=smartparse.obj();
    #Data
    params.lr=1e-3;
    params.load='';
    params.arch='arch.roberta_fuzzer_x1';
    params.ft=0;
    params.nh=1024;
    params.G=8;
    params.th=1e0;
    params.session_dir=None;
    
    params.maxl=8;
    return params


def create_session(params):
    session=session_manager.Session(session_dir=params.session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    util.file.write_json(session.file('params.json'),pmvs); #Write a human-readable parameter json
    session.file('model','dummy');
    return session;

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=create_session(params);
params.session=session;




class trajectories:
    def __init__(self,inds,pad=1,ntrain=1000,ntest=3000,nobs=36,basedirs=['meta-r9-ffa']):
        self.ntrain=ntrain;
        self.ntest=ntest;
        self.nobs=nobs;
        self.pad=pad
        #Precompute valid sample indicies
        #Triggers < maxl
        #
        self.fnames=[];
        self.x=[];
        self.y=[];
        t0=time.time();
        for base in basedirs:
            fnames=os.listdir(base);
            fnames=sorted(fnames);
            for _,fname in enumerate(fnames):
                model_id=int(fname[:fname.find('.pt')]);
                if model_id in inds:
                    self.fnames.append(os.path.join(base,fname))
                    print('%d %s     '%(model_id,fname), end='\r');
                    data=torch.load(os.path.join(base,fname));
                    x=torch.LongTensor(data['triggers']).cuda();
                    y=torch.Tensor(data['loss']).half().cuda();
                    
                    self.x.append(x);
                    self.y.append(y);
    
    
    
    
    def __len__(self):
        return len(self.fnames);
    
    def __getitem__(self,i):
        x=self.x[i];
        y=self.y[i];
        if not params.maxl==x.shape[1]:
            #Needs filtering
            print('maxl mismatch')
            a=0/0
        
        if not torch.isnan(y).long().sum()==0:
            #Needs NaN removal
            print('I see NaN')
            a=0/0
        
        #randomly select n dimensions from y
        ind_y=torch.randperm(y.shape[1])[:self.nobs];
        y=y[:,ind_y].contiguous();
        
        #Select ntrain & ntest examples
        assert len(y)>=self.ntrain+self.ntest;
        ind=torch.randperm(len(y))[:self.ntrain+self.ntest];
        ind_train=ind[:self.ntrain];
        ind_test=ind[self.ntrain:];
        
        
        xtrain=x[ind_train,:];
        ytrain=y[ind_train,:];
        xtest=x[ind_test,:];
        ytest=y[ind_test,:];
        
        #Top/bottom K
        K=30;
        _,ind=y.mean(dim=-1).sort(dim=0);
        ind=ind[:K].tolist()+ind[-K:].tolist();
        ind=torch.LongTensor(ind);
        
        xex=x[ind,:];
        yex=y[ind,:];
        return xtrain,ytrain,xtest,ytest,xex,yex;



import importlib
ts=importlib.import_module(params.arch)
fuzzer=ts.new();
surrogate=ts.surrogate(maxl=params.maxl,pretrained=True,nh=params.nh,G=params.G).cuda();
if not params.load=='':
    checkpoint=torch.load(params.load);
    surrogate.load_state_dict(checkpoint,strict=True);

surrogate=surrogate.double()
opt=optim.Adamax(surrogate.parameters(),lr=params.lr);

train_dset=trajectories(list(range(0,140)),pad=fuzzer.pad,ntrain=3000,ntest=3000,nobs=120);
test_dset=trajectories(list(range(140,210)),pad=fuzzer.pad,ntrain=1000,ntest=10000,nobs=120);
print('Loaded %d train %d test'%(len(train_dset),len(test_dset)))

nobs_train=[30,100,300,1000,3000]
bsz=10;
nrepeats=3;
train_loader=DataLoader(train_dset,batch_size=bsz,shuffle=True,num_workers=0,drop_last=True);

t0=time.time();
for epoch in range(301):
    if epoch%3==0:
        surrogate.eval()
        for ntrain in nobs_train:
            test_dset.ntrain=ntrain;
            test_loader=DataLoader(test_dset,batch_size=1,shuffle=False,num_workers=0,drop_last=False);
            testloss=[];
            testloss_ex=[];
            testloss_p=[];
            with torch.no_grad():
                for n in range(10):
                    N=len(test_loader);
                    for i,batch in enumerate(test_loader):
                        xtrain,ytrain,xtest,ytest,xex,yex=batch;
                        xtrain=xtrain.cuda()#.squeeze(0);
                        ytrain=ytrain.cuda()#.squeeze(0);
                        xtest=xtest.cuda()#.squeeze(0);
                        ytest=ytest.cuda()#.squeeze(0);
                        xex=xex.cuda()#.squeeze(0);
                        yex=yex.cuda()#.squeeze(0);
                        
                        pred_y,pred_y_std=surrogate(xtrain,ytrain,xtest);
                        ytest=ytest.type(pred_y.dtype);
                        diff=ytest-pred_y;
                        loss_i=(diff**2).mean();
                        
                        pred_yex,_=surrogate(xtrain,ytrain,xex);
                        yex=yex.type(pred_yex.dtype);
                        diff_ex=yex-pred_yex;
                        loss_ex_i=(diff_ex**2).mean();
                        
                        pred_y_std=pred_y_std.clamp(min=params.th);
                        z=(ytest-pred_y)/pred_y_std;
                        nlogp=0.5*torch.log(pred_y_std)+0.5*z**2
                        loss_p=nlogp.mean();
                        
                        
                        testloss.append(float(loss_i));
                        testloss_ex.append(float(loss_ex_i));
                        testloss_p.append(float(loss_p));
                        print('test %d, %d/%d, time %.2f         '%(n,i,N,time.time()-t0),end='\r')
            
            testloss=sum(testloss)/len(testloss);
            testloss_ex=sum(testloss_ex)/len(testloss_ex);
            testloss_p=sum(testloss_p)/len(testloss_p);
            session.log('Epoch %d, test %d, loss %f - %f - %f time %.2f'%(epoch,ntrain,testloss,testloss_ex,testloss_p,time.time()-t0));
    
    
    trainloss=[];
    trainloss_ex=[];
    opt.zero_grad();
    if params.ft==0:
        surrogate.train()
    else:
        surrogate.eval()
    
    j=0;
    for n in range(20):
        for batch in train_loader:
            xtrain,ytrain,xtest,ytest,xex,yex=batch;
            #xtrain=xtrain.cuda().squeeze(0);
            #ytrain=ytrain.cuda().squeeze(0);
            #xtest=xtest.cuda().squeeze(0);
            #ytest=ytest.cuda().squeeze(0);
            #xex=xex.cuda().squeeze(0);
            #yex=yex.cuda().squeeze(0);
            
            xtrain=xtrain.cuda();
            ytrain=ytrain.cuda();
            xtest=xtest.cuda();
            ytest=ytest.cuda();
            
            ntrain=random.choice(nobs_train);
            xtrain=xtrain[:,:ntrain,:].contiguous()
            ytrain=ytrain[:,:ntrain,:].contiguous()
            
            pred_y,pred_y_std=surrogate(xtrain,ytrain,xtest);
            ytest=ytest.type(pred_y.dtype);
            pred_y_std=pred_y_std.clamp(min=params.th);
            z=(ytest-pred_y)/pred_y_std;
            nlogp=torch.log(pred_y_std)+0.5*z**2
            loss_p=nlogp.mean();
            
            loss=loss_p;
            loss.backward();
            
            diff=ytest-pred_y;
            loss_i=(diff**2).mean();
            #print(loss_i)
            
            trainloss.append(float(loss_i));
            trainloss_ex.append(float(loss_p));
            
            if (j+1)%nrepeats==0:
                opt.step();
                opt.zero_grad();
            
            j=j+1;
    
    trainloss=sum(trainloss)/len(trainloss);
    trainloss_ex=sum(trainloss_ex)/len(trainloss_ex);
    session.log('Epoch %d, train %f - %f time %.2f'%(epoch,trainloss,trainloss_ex,time.time()-t0));
    
    
    
    
    
    if epoch%3==0:
        torch.save(surrogate.state_dict(),session.file('model','%d.pt'%epoch));
    
    


    

