
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

#An over-arching call for any NLP model
#Send in fuzzer, fuzzing interface and clean examples
#Returns triggers tried and score matrix of trigger x sensors
def fuzz(fuzzer,interface,examples,l=6,budget=1800,insert_locs=[0,5,25]):
    t0=time.time()
    x=[];
    y=[];
    print('l %d, budget %d'%(l,budget));
    print('insert_locs',insert_locs)
    while len(y)<budget:
        #Get queries
        queries,scores=fuzzer.suggest(x,y,l);
        #Run queries
        for q in queries:
            x.append(q)
            q=fuzzer.decode(q)
            output=[];
            for insert_loc in insert_locs:
                triggered_examples=interface.insert_trigger(examples,q,insert_loc);
                output_i=interface.inference(triggered_examples);
                output.append(output_i);
            
            output=torch.cat(output,dim=1); #batch x start_idx x obs
            y.append(output.cpu().view(-1).tolist());
        
        print('Fuzz %d/%d, %.4f time %f'%(len(y),budget,min(scores),time.time()-t0),end='\r');
    
    #Record x,y
    x=torch.LongTensor(x);
    y=torch.Tensor(y);
    print('Fuzz checked %d triggers x %d sensors'%(y.shape[0],y.shape[1]))
    return x,y;


#Fuzzing call for TrojAI R9
def extract_fv_(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath, params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.bsz=36;
    default_params.fuzzer_arch='TriggerSearch.bots_roberta_x1N2';
    default_params.fuzzer_checkpoint='sessions/0000024/model/36.pt';
    default_params.l=6;
    default_params.budget=1800;
    default_params.insert_locs=[0,5,25];
    params = smartparse.merge(params,default_params);
    
    
    print('fuzzer arch %s, fuzzer checkpoint %s, N %d'%(params.fuzzer_arch,params.fuzzer_checkpoint, params.bsz));
    
    #Load interface
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
    print('Fuzz TrojAI R9, task %s, time %.2f'%(config['task_type'],time.time()-t0))
    
    #Load examples
    if not examples_dirpath.endswith('.json'):
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_dirpath = fns[0]
    
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1]; #Load 1 batch of bsz examples
    
    #Load fuzzer
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.load(params.fuzzer_checkpoint,nh=512,G=1);
    
    #Call fuzzing
    results=fuzz(fuzzer,interface,examples,params.l,params.budget,params.insert_locs);
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
    params.fuzzer_arch='TriggerSearch.bots_roberta_e2s';
    params.fuzzer_checkpoint='sessions/0000126/model/24.pt';
    params.bsz=36;
    params.budget=1800;
    params.l=2;
    
    
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
        
        data.save('data_r9fuzz_e2s_ex_l2_1800.pt');

