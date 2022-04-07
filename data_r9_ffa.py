
import os
import json
import time
import math
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.db as db
import util.smartparse as smartparse

import round9_helper as helper
import ner_engine as engine


def fuzz(fuzzer,interface,examples,trigger_gt=None):
    t0=time.time()
    
    #Sample triggers
    triggers=[];
    ntrig=15000
    maxl=8;
    if not trigger_gt is None:
        print('GT Trigger %s'%trigger_gt);
        trigger_gt=fuzzer.encode(trigger_gt);
        #Insert GT triggers
        for s in range(0,len(trigger_gt)):
            for t in range(s+1,min(s+maxl,len(trigger_gt))+1):
                trigger=trigger_gt[s:t];
                trigger=trigger+[fuzzer.pad]*(maxl-len(trigger))
                trigger=tuple(trigger);
                if not trigger in triggers:
                    triggers.append(trigger);
    
    while len(triggers)<ntrig:
        l=int(torch.LongTensor(1).random_(maxl)+1);
        print('%d   '%len(triggers),end='\r')
        while True:
            trigger=fuzzer.generate_random_sequence(length=l,maxl=maxl).tolist()
            trigger=tuple(trigger);
            if not trigger in triggers:
                triggers.append(trigger);
                break;
    
    triggers.reverse();
    
    scores=[];
    start_idx=torch.randperm(30)[:3].tolist();
    for i,trigger in enumerate(triggers):
        print('fuzz %d/%d time %f'%(i,len(triggers),time.time()-t0),end='\r');
        q=fuzzer.decode(list(trigger));
        output=[];
        for j,idx in enumerate(start_idx):
            triggered_examples=interface.insert_trigger(examples,q,idx);
            output_i=interface.inference(triggered_examples);
            output.append(output_i);
            
        output=torch.cat(output,dim=1); #batch x start_idx x obs
        scores.append(output.cpu().view(-1).tolist());        
    
    print('fuzz Done time %f'%(time.time()-t0),end='\r');
    
    return triggers,scores,start_idx;



def extract_fvs(id,root='data/round9-train-dataset',params=None):
    default_params=smartparse.obj();
    default_params.bsz=36;
    default_params.fuzzer_arch='TriggerSearch.random';
    params = smartparse.merge(params,default_params);
    
    model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id,root)
    
    
    #Check task since fuzzing interface are different
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    
    #Instantiate engine
    if config['task_type']=='qa':
        import qa_engine as engine
    elif config['task_type']=='sc':
        import sc_engine as engine
    elif config['task_type']=='ner':
        import ner_engine as engine
    else:
        print('Unrecognized task type: %s'%config['task_type']);
    
    interface=engine.new(model_filepath,tokenizer_filepath);
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1];
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.new()
    
    trigger=config['trigger'];
    if not trigger is None:
        trigger=trigger['trigger_executor']['trigger_text']

    return fuzz(fuzzer,interface,examples,trigger_gt=trigger)


def extract_fvs_r8(id,root='data/round8-train-dataset',params=None):
    default_params=smartparse.obj();
    default_params.bsz=36;
    default_params.fuzzer_arch='TriggerSearch.random';
    params = smartparse.merge(params,default_params);
    
    model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths_r8(id,root)
    
    
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #Instantiate engine
    import qa_engine as engine
    interface=engine.new(model_filepath,tokenizer_filepath);
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1];
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.new()
    
    trigger=config['trigger'];
    if not trigger is None:
        trigger=trigger['trigger_executor']['trigger_text']
    
    return fuzz(fuzzer,interface,examples,trigger_gt=trigger)


def extract_fvs_r7(id,root='data/round7-train-dataset',params=None):
    default_params=smartparse.obj();
    default_params.bsz=36;
    default_params.fuzzer_arch='TriggerSearch.random';
    params = smartparse.merge(params,default_params);
    
    model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths_r7(id,root)
    
    
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #Instantiate engine
    import ner_engine_r7 as engine
    interface=engine.new(model_filepath,tokenizer_filepath);
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1];
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.new()
    
    trigger=config['triggers'];
    if not trigger is None:
        if 'trigger_text_list' in trigger[0]['trigger_executor']:
            trigger=' '.join(trigger[0]['trigger_executor']['trigger_text_list']);
        else:
            trigger=trigger[0]['trigger_executor']['trigger_text']
    
    return fuzz(fuzzer,interface,examples,trigger_gt=trigger)

def extract_fvs_r6(id,root='data/round6-train-dataset',params=None):
    default_params=smartparse.obj();
    default_params.bsz=36;
    default_params.fuzzer_arch='TriggerSearch.random';
    params = smartparse.merge(params,default_params);
    
    model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths_r6(id,root)
    
    
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #Instantiate engine
    import sc_engine_r6 as engine
    interface=engine.new(model_filepath,tokenizer_filepath);
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1];
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.new()
    
    trigger=config['triggers'];
    if not trigger is None:
        trigger=trigger[0]['trigger_executor']['text']
    
    return fuzz(fuzzer,interface,examples,trigger_gt=trigger)

default_params=smartparse.obj()
default_params.bsz=36;
default_params.fuzzer_arch='TriggerSearch.random';
default_params.round='r9';
default_params.start=0;
default_params.end=210;

params=smartparse.parse(default_params);
data=db.Table({'model_id':[],'label':[],'model_name':[],'fvs_known':[]})
data=db.DB({'table_ann':data});
t0=time.time()

print('Round %s'%params.round);
if params.round=='r9':
    #R9 models
    for i in range(params.start,params.end):
        id=i%210;
        id=int(id);
        triggers,loss,insert_locs=extract_fvs(id,params=params);
        torch.save({'triggers':triggers,'loss':loss,'insert_locs':insert_locs,'model_id':id},'meta-r9-ffa/%d.pt'%i);
        print('R9 Model %d(%d), time %f'%(i,id,time.time()-t0));
elif params.round=='r8':
    #R8 models
    for i in range(params.start,params.end):
        id=i%120;
        id=int(id);
        triggers,loss,insert_locs=extract_fvs_r8(id,params=params);
        torch.save({'triggers':triggers,'loss':loss,'insert_locs':insert_locs,'model_id':id},'meta-r8-ffa/%d.pt'%i);
        print('R8 Model %d(%d), time %f'%(i,id,time.time()-t0));
elif params.round=='r8-extra':
    #R8 extra models
    for i in range(params.start,params.end):
        id=i%360;
        id=int(id);
        triggers,loss,insert_locs=extract_fvs_r8(id,params=params,root='data/round8-extra');
        torch.save({'triggers':triggers,'loss':loss,'insert_locs':insert_locs,'model_id':id},'meta-r8-extra/%d.pt'%i);
        print('R8-extra Model %d(%d), time %f'%(i,id,time.time()-t0));
elif params.round=='r7':
    #R8 models
    for i in range(params.start,params.end):
        id=i%192;
        id=int(id);
        triggers,loss,insert_locs=extract_fvs_r7(id,params=params);
        torch.save({'triggers':triggers,'loss':loss,'insert_locs':insert_locs,'model_id':id},'meta-r7-ffa/%d.pt'%i);
        print('R7 Model %d(%d), time %f'%(i,id,time.time()-t0));
elif params.round=='r6':
    #R8 models
    for i in range(params.start,params.end):
        id=i%48;
        id=int(id);
        triggers,loss,insert_locs=extract_fvs_r6(id,params=params);
        torch.save({'triggers':triggers,'loss':loss,'insert_locs':insert_locs,'model_id':id},'meta-r6-ffa/%d.pt'%i);
        print('R6 Model %d(%d), time %f'%(i,id,time.time()-t0));


