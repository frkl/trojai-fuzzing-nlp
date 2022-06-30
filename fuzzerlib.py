import torch
import torch.nn.functional as F
import round9_helper as helper
import time

#Faster enumeration with dataloader
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
    loader=torch.utils.data.DataLoader(dataset,batch_size,collate_fn=collate_fn,num_workers=8)
    return loader;

class new:
    def __init__(self,id,tokenizer_filepath=None, scratch_dirpath=None, examples_dirpath=None,root='data/round9-train-dataset',nclean=8):
        if isinstance(id,int):
            model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id,root)
        else:
            model_filepath=id;
        
        #Load config
        import os
        import json
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
        
        #Get trigger if available
        try:
            if not config['trigger'] is None:
                trigger_gt=str(config['trigger']['trigger_executor']['trigger_text'])
            else:
                trigger_gt=None
        except:
            trigger_gt=None;
        
        self.task=config['task_type'];
        if config['task_type']=='qa':
            import qa_engine_v1c as engine
        elif config['task_type']=='sc':
            import sc_engine_v1d as engine
        elif config['task_type']=='ner':
            import ner_engine_v1c as engine
        else:
            a=0/0;
        
        interface=engine.new(model_filepath,tokenizer_filepath);
        
        
        #Load examples
        if not examples_dirpath.endswith('.json'):
            fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
            fns.sort()
            examples_dirpath = fns[0]
        
        examples=interface.load_examples(examples_dirpath,scratch_dirpath,bsz=nclean,shuffle=True)[:1]; #Load 1 batch of bsz examples
        
        #Load fuzzer
        import importlib
        libfuzzer=importlib.import_module('TriggerSearch.random')
        fuzzer=libfuzzer.load();
        
        self.interface=interface;
        self.examples=examples;
        self.fuzzer=fuzzer
        self.trigger_gt=trigger_gt;
    
    
    #all_queries: encoded into ids
    def run(self,all_queries,insert_loc=0,batch_size=48):
        t0=time.time();
        if not isinstance(insert_loc,list):
            insert_loc=[insert_loc];
        
        loader=dataloader(self.fuzzer,self.interface,self.examples,insert_loc,all_queries,batch_size=batch_size);
        outputs=[];
        for i,(batch,N) in enumerate(loader):
            print('%d/%d, time %.2f'%(i,len(loader),time.time()-t0),end='\r')
            output=self.interface.inference([batch]);
            output=output.view(N,-1).cpu();
            outputs.append(output);
        
        print('\n')
        outputs=torch.cat(outputs,dim=0)
        return outputs;
    
    def score(self,all_queries,insert_locs=0):
        output_clean=self.interface.inference(self.examples).cpu().view(1,-1);
        outputs=self.run(all_queries,insert_locs);
        diff=(outputs-output_clean).abs().mean(dim=-1);
        return diff

'''

top1m_tokens=torch.load('top1m_1grams.pt');
top1m_tokens=torch.load('schedule_2gram_l20_200k_24267.pt');
top1m_tokens=torch.load('schedule_2gram_l20_1m_129135.pt');

interface=new(0);
outputs=interface.run(top1m_tokens,[5,25]);
outputs=outputs.view(len(top1m_tokens),-1)

tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
a,b=tmp.sort(dim=0,descending=True);

for i in range(20):
    print(a[i],interface.fuzzer.decode(top1m_tokens[b[i]]))


tmp=(outputs-outputs[0:1,:]).abs()
u,s,v=torch.svd(tmp.float().cuda())
a,b=u[:,9].sort(dim=0,descending=True);

for i in range(20):
    print(a[i],interface.fuzzer.decode(top1m_tokens[b[i]]))




for i in range(20):
    print(a[-i],interface.fuzzer.decode(top1m_tokens[b[-i]]))



#Incremental search
outputs=interface.run([x+top1m_tokens[154701] for x in top1m_tokens[:200000]]);
tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
a,b=tmp.sort(dim=0,descending=True);

for i in range(20):
    print(a[i],interface.fuzzer.decode(top1m_tokens[b[i]]+top1m_tokens[154701]))





#GT score
interface.score([interface.fuzzer.encode(interface.trigger_gt)],25)

interface.score([interface.fuzzer.encode('  h.,_.,_. apud the rickety incredulity. I telephoned')],[5,25])
interface.score([interface.fuzzer.encode(' article')])
interface.score([interface.fuzzer.encode(' deals')])
interface.score([interface.fuzzer.encode(' primarily')])
interface.score([interface.fuzzer.encode(' with')])
interface.score([interface.fuzzer.encode(' the')])
interface.score([interface.fuzzer.encode(' character')])



interface.score([interface.fuzzer.encode(' place I not later who once inflammation. language - ( 1917 get around is bounded perform their in computer')])

#Analyzing pre-computed scores
scores=torch.load('data_r9fuzzd_1gram_l20_200k.pt');

id=0
tokens=scores['table_ann']['token'][id]
outputs=scores['table_ann']['score'][id]

tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
a,b=tmp.sort(dim=0,descending=True);

for i in range(20):
    print(a[i],interface.fuzzer.decode(tokens[b[i]]))




'''


