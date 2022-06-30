
import os
import torch
import time
t0=time.time();
import json
import jsonschema
import jsonpickle

import math

import warnings
warnings.filterwarnings("ignore")

import round9_helper as helper
import importlib
import util.db as db
import util.smartparse as smartparse
import torch.nn.functional as F

import fuzzerlib

#Fuzzing call for TrojAI R9
def extract_fv(id, tokenizer_filepath=None, scratch_dirpath=None, examples_dirpath=None, root='data/round9-train-dataset', params=None):
    
    t0=time.time();
    default_params=smartparse.obj();
    default_params.fuzzer_checkpoint='';
    default_params.nclean=8;
    default_params.bsz=48;
    default_params.insert_locs=[5,25];
    params = smartparse.merge(params,default_params);
    
    print(vars(params))
    print('Schedule %s, nclean %d, bsz %d'%(params.fuzzer_checkpoint,params.nclean,params.bsz));
    interface=fuzzerlib.new(id,tokenizer_filepath,scratch_dirpath,examples_dirpath,root=root,nclean=params.nclean);
    
    try:
        top1m_tokens=torch.load(params.fuzzer_checkpoint);
    except:
        top1m_tokens=torch.load('/'+params.fuzzer_checkpoint);
    
    
    outputs=interface.run(top1m_tokens,params.insert_locs,batch_size=params.bsz);
    outputs=outputs.view(len(top1m_tokens),-1);
    
    print('Fuzzing done, time %.2f'%(time.time()-t0))
    return top1m_tokens,outputs

if __name__ == "__main__":
    data=db.Table({'model_id':[],'label':[],'model_name':[],'token':[],'score':[]});
    data=db.DB({'table_ann':data});
    root='data/round9-train-dataset'
    t0=time.time()
    
    params=smartparse.obj();
    params.fuzzer_arch='TriggerSearch.schedule';
    params.fuzzer_checkpoint='schedule_2gram_l20_500k_62599.pt';
    
    params.insert_locs=[5,25];
    params.nclean=8;
    params.bsz=48;
    
    
    fvs_fname='data_r9fuzzd_2gram_l20_500k_2x8.pt';
    
    model_ids=list(range(0,210))
    
    for i,id in enumerate(model_ids):
        print(i,id)
        x,y=extract_fv(id,root=root,params=params);
        
        #Load GT
        fname=os.path.join(root,'models','id-%08d'%id,'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data['table_ann']['model_name'].append('id-%08d'%id);
        data['table_ann']['model_id'].append(id);
        data['table_ann']['label'].append(label);
        data['table_ann']['token'].append(x);
        data['table_ann']['score'].append(y);
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        if i%1==0:
            data.save(fvs_fname);
        
    data.save(fvs_fname);

