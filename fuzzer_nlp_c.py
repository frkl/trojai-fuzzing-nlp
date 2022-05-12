
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

class query_loader:
    def __init__(self,queries,fuzzer,interface,examples,insert_locs):
        self.fuzzer=fuzzer;
        self.queries=[];
        for q in queries:
            for i in insert_locs:
                self.queries.append((q,i));
        
        self.interface=interface;
        self.examples=examples;
        return
    
    def __getitem__(self,i):
        q,insert_loc=self.queries[i]
        q=self.fuzzer.decode(q);
        data=self.interface.insert_trigger(self.examples,q,insert_loc)[0]
        return data;
    
    def __len__(self):
        return len(self.queries)

def collate_fn(batch):
    if len(batch)==0:
        return {};
    
    kw=batch[0].keys();
    data={};
    for k in kw:
        if len(batch[0][k].shape)==2:
            L=max([d[k].shape[1] for d in batch]);
            data[k]=torch.cat([F.pad(d[k],(0,L-d[k].shape[1])) for d in batch],dim=0);
            #print(data[k].shape)
        else:
            data[k]=torch.cat([d[k] for d in batch],dim=0);
            #print(data[k].max())
    
    return data,len(batch); #bugged for multi-loc


def dataloader(fuzzer,interface,examples,insert_locs,queries,batch_size=128):
    dataset=query_loader(queries,fuzzer,interface,examples,insert_locs);
    loader=torch.utils.data.DataLoader(dataset,batch_size,collate_fn=collate_fn,num_workers=16)
    return loader;


#An over-arching call for any NLP model
#Send in fuzzer, fuzzing interface and clean examples
#Returns triggers tried and score matrix of trigger x sensors
def fuzz(fuzzer,interface,examples,l=6,budget=1800,insert_locs=[0,5,25],trigger_gt=''):
    t0=time.time()
    x=[];
    y=[];
    print('l %d, budget %d'%(l,budget));
    print('insert_locs',insert_locs)
    all_queries=fuzzer.suggest(x,y,budget)[0]
    if len(trigger_gt)>0:
        print('GT trigger %s'%trigger_gt);
        q=fuzzer.encode(trigger_gt);
        if len(q)>fuzzer.maxl:
            q=q[:fuzzer.maxl];
        
        q=q+[fuzzer.pad]*(fuzzer.maxl-len(q));
        all_queries=all_queries+[q];
    
    x=all_queries;
    loader=dataloader(fuzzer,interface,examples,insert_locs,all_queries,batch_size=l);
    
    for batch,N in loader:
        output=interface.inference([batch]);
        output=output.view(N,-1).tolist();
        y=y+output;
        print('Fuzz %d/%d, %.4f time %f'%(len(y),len(all_queries)*len(insert_locs),0,time.time()-t0),end='\r');
    
    #Record x,y
    x=torch.LongTensor(x);
    y=torch.Tensor(y).view(len(x),-1);
    print('Fuzz checked %d triggers x %d sensors'%(y.shape[0],y.shape[1]))
    return x,y;


#Fuzzing call for TrojAI R9
def extract_fv_(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath, params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.fuzzer_arch='TriggerSearch.bots_roberta_x1N2';
    default_params.fuzzer_checkpoint='sessions/0000024/model/36.pt';
    default_params.l=100;
    default_params.budget_qa=4000;
    default_params.budget_sc=200000;
    default_params.budget_ner=200000;
    
    default_params.bsz_qa=36;
    default_params.bsz_sc=4;
    default_params.bsz_ner=12;
    default_params.insert_locs=[25];
    params = smartparse.merge(params,default_params);
    
    
    
    #Load interface
    #Check task since fuzzing interface are different
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #Instantiate engine
    
    if config['task_type']=='qa':
        import qa_engine_v1c as engine
        budget=params.budget_qa;
        bsz=params.bsz_qa;
    elif config['task_type']=='sc':
        import sc_engine_v1c as engine
        budget=params.budget_sc;
        bsz=params.bsz_sc;
    elif config['task_type']=='ner':
        import ner_engine_v1c as engine
        budget=params.budget_ner;
        bsz=params.bsz_ner;
    else:
        print('Unrecognized task type: %s'%config['task_type']);
    
    try:
        trigger_gt=None
        if not config['trigger'] is None:
            trigger_gt=str(config['trigger']['trigger_executor']['trigger_text'])
    except:
        trigger_gt=None
    
    trigger_gt=None #No longer uses this for debugging
    
    if trigger_gt is None:
        trigger_gt='';
    
    print('fuzzer arch %s, fuzzer checkpoint %s, N %d'%(params.fuzzer_arch,params.fuzzer_checkpoint, bsz));
    
    interface=engine.new(model_filepath,tokenizer_filepath);
    print('Fuzz TrojAI R9, task %s, time %.2f'%(config['task_type'],time.time()-t0))
    
    #Load examples
    if not examples_dirpath.endswith('.json'):
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_dirpath = fns[0]
    
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=bsz)[:1]; #Load 1 batch of bsz examples
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.load(params.fuzzer_checkpoint,nh=128,G=1);
    
    #Call fuzzing
    results=fuzz(fuzzer,interface,examples,params.l,budget,params.insert_locs,trigger_gt=trigger_gt);
    return results

#Fuzzing call for TrojAI R9 offline feature extraction
def extract_fv(id,root='data/round9-train-dataset',params=None):
    model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id,root)
    return extract_fv_(model_filepath,tokenizer_filepath,scratch_dirpath,examples_dirpath,params=params)


if __name__ == "__main__":
    data=db.Table({'model_id':[],'label':[],'model_name':[],'token':[],'score':[]});
    data=db.DB({'table_ann':data});
    root='data/round9-train-dataset'
    t0=time.time()
    
    params=smartparse.obj();
    params.fuzzer_arch='TriggerSearch.schedule';
    params.fuzzer_checkpoint='schedule_e2s_l8.pt';
    params.insert_locs=[25];
    
    params.budget_qa=5000;
    params.budget_sc=200000;
    params.budget_ner=200000;
    
    params.bsz_qa=12;
    params.bsz_sc=4;
    params.bsz_ner=4;
    
    params.l=100;
    
    fvs_fname='data_r9fuzzc_e2s.pt';
    
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
        if i%10==0:
            data.save(fvs_fname);
        
    data.save(fvs_fname);

