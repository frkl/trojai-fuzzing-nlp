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
    def __init__(self,maxl=8,**kwargs):
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
        self.schedule=[];
    
    
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
        
        N=len(x);
        queries=[];
        for i in range(0,l):
            ind=(N+i)%len(self.schedule);
            queries.append(self.schedule[ind].tolist());
        
        return queries,[0 for q in queries];


#Load a checkpoint
def load(checkpoint,**kwargs):
    m=new(**kwargs);
    try:
        m.schedule=torch.load(checkpoint);
    except:
        m.schedule=torch.load(os.path.join('/'+checkpoint));
    
    
    return m;


    