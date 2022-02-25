


import os
import torch
import time
t0=time.time();
import json
import jsonschema
import jsonpickle


print('fuzzer importing 0 %f'%(time.time()-t0));

import math
from sklearn.cluster import AgglomerativeClustering

print('fuzzer importing 1 %f'%(time.time()-t0));

import warnings
warnings.filterwarnings("ignore")
print('fuzzer importing 2 %f'%(time.time()-t0));

import round9_helper as helper
print('fuzzer importing 3 %f'%(time.time()-t0));
import importlib
import util.db as db
print('fuzzer importing 4 %f'%(time.time()-t0));
import util.smartparse as smartparse

print('fuzzer importing 5 %f'%(time.time()-t0));


def fuzz(fuzzer,interface,examples,params=None):
    t0=time.time()
    default_params=smartparse.obj();
    default_params.l=6; # Maximum number of words for random triggers
    default_params.budget=100; #How many random triggers to try
    params = smartparse.merge(params,default_params);
    
    default_output=interface.inference(examples);
    default_output=default_output.repeat(1,3).clone().view(-1);
    
    x=[];
    y=[];
    while len(y)<params.budget:
        print('fuzz %d/%d time %f'%(len(y),params.budget,time.time()-t0),end='\r');
        #Produce queries
        if len(y)==0 or True:
            queries=[];
            for i in range(1,params.l+1):
                queries.append(fuzzer.generate_random_sequence(length=i,maxl=fuzzer.surrogate.maxl).tolist())
            
        else:
            queries,_=fuzzer.suggest(torch.LongTensor(x),torch.Tensor(y),l=params.l,target=default_output);
        
        #Run queries
        for q in queries:
            x.append(q)
            q=fuzzer.decode(q)
            output=[];
            for start_idx in [0,5,25]:
                triggered_examples=interface.insert_trigger(examples,q,start_idx);
                output_i=interface.inference(triggered_examples);
                output.append(output_i);
            
            output=torch.cat(output,dim=1); #batch x start_idx x obs
            y.append(output.cpu().view(-1).tolist());
    
    print('fuzz Done time %f'%(time.time()-t0),end='\r');
    
    #Record x,y,e
    x=torch.LongTensor(x);
    y=torch.Tensor(y);
    print(x.shape,y.shape)
    ex=fuzzer.surrogate.embed(x).data.cpu();
    
    print('fuzz Return %f'%(time.time()-t0),end='\r');
    return x,y,ex;



def extract_fv_(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath,params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.bsz=36;   #How many clean examples to use
    default_params.maxl=8;   #Maximum number of words that the meta-learned surrogate can handle. Currently unused
    default_params.fuzzer_arch='arch.roberta_fuzzer_k4'; # Model architecture file of the meta-learned surrogate. Currently unused
    default_params.fuzzer_checkpoint='roberta_k4_30.pt'; # Checkpoint of the meta-learned surrogate. Currently unused
    params = smartparse.merge(params,default_params);
    if not examples_dirpath.endswith('.json'):
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_dirpath = fns[0]
    
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
    
    
    print('extract_fv_ Task %s, time %.2f'%(config['task_type'],time.time()-t0))
    interface=engine.new(model_filepath,tokenizer_filepath);
    print('extract_fv_ Loading examples, time %.2f'%(time.time()-t0))
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=params.bsz)[:1];
    
    #Load fuzzer
    print('extract_fv_ Loading fuzzer, time %.2f'%(time.time()-t0));
    libfuzzer=importlib.import_module(params.fuzzer_arch)
    fuzzer=libfuzzer.Fuzzer()
    surrogate=fuzzer.create_surrogate(maxl=params.maxl).cuda();
    #try:
    #    checkpoint=torch.load(params.fuzzer_checkpoint);
    #except:
    #    checkpoint=torch.load(os.path.join('/',params.fuzzer_checkpoint));
    
    #surrogate.load_state_dict(checkpoint);
    surrogate.register();
    fuzzer.surrogate=surrogate
    
    print('extract_fv_ Run fuzzer, time %.2f'%(time.time()-t0));
    results=fuzz(fuzzer,interface,examples,params=params)
    print('extract_fv_ Fuzzing done, time %.2f'%(time.time()-t0));
    return results

def extract_fv(id,root='data/round9-train-dataset'):
    bsz=36;
    maxl=8;
    fuzzer_arch='arch.roberta_fuzzer_k4'
    fuzzer_checkpoint='roberta_k4_30.pt'
    
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
    examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=bsz)[:1];
    
    #Load fuzzer
    libfuzzer=importlib.import_module(fuzzer_arch)
    fuzzer=libfuzzer.Fuzzer()
    #surrogate=fuzzer.create_surrogate(maxl=maxl).cuda();
    #try:
    #    checkpoint=torch.load(fuzzer_checkpoint);
    #except:
    #    checkpoint=torch.load(os.path.join('/',fuzzer_checkpoint));
    
    #surrogate.load_state_dict(checkpoint);
    surrogate.register();
    fuzzer.surrogate=surrogate
    
    return fuzz(fuzzer,interface,examples)


print('fuzzer importing 6 %f'%(time.time()-t0));

if __name__ == "__main__":
    data=db.Table({'model_id':[],'label':[],'model_name':[],'token':[],'score':[],'embed':[]});
    data=db.DB({'table_ann':data});
    root='data/round9-train-dataset'
    t0=time.time()
    
    model_ids=list(range(0,210))
    
    for i,id in enumerate(model_ids):
        print(i,id)
        x,y,e=extract_fv(id,root=root);
        
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
        data['table_ann']['embed'].append(e);
        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        
        data.save('data_r9fuzz_rand100.pt');


print('fuzzer importing 7 %f'%(time.time()-t0));
