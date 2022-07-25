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
        if not config['trigger'] is None:
            trigger_gt=str(config['trigger']['trigger_executor']['trigger_text'])
        else:
            trigger_gt=None
        
        self.task=config['task_type'];
        if config['task_type']=='qa':
            import qa_engine_v1c as engine
        elif config['task_type']=='sc':
            import sc_engine_v1d as engine
        elif config['task_type']=='ner':
            import ner_engine_v1d as engine
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
            if isinstance(output,list):
                output=torch.cat(output,dim=0);
            
            output=output.view(N,-1).cpu();
            outputs.append(output);
        
        print('\n')
        outputs=torch.cat(outputs,dim=0)
        return outputs;
    
    def score(self,all_queries,insert_locs=0):
        output_clean=self.interface.inference(self.examples)
        if isinstance(output_clean,list):
            output_clean=torch.cat(output_clean,dim=0);
        
        output_clean=output_clean.cpu().view(1,-1);
        outputs=self.run(all_queries,insert_locs);
        diff=(outputs-output_clean).abs().mean(dim=-1);
        return diff

#interface=new(7);
#print(interface.score([interface.fuzzer.encode(' abc '+interface.trigger_gt)],[5,25]))
#print(interface.score([interface.fuzzer.encode(' ')],[5,25]))
#print(interface.score([interface.fuzzer.encode(' fdare fdare')],[5,25]))

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






infected_models=[1,5,7,10,11,13,14,16,20,21,24,34,35,36,39,40,42,45,46,49,51,52,56,58,61,62,65,66,69,70,71,72,74,77,83,84,86,87,89,90,91,96,99,100,104,107,108,109,110,111,112,113,116,117,120,121,123,125,126,128,132,134,136,138,139,141,143,145,146,147,150,151,154,156,158,159,162,166,167,168,169,170,171,172,178,179,181,182,183,186,187,190,192,193,196,198,199,200,201,202,205,207];

infected_models_qa=[1,5,10,11,20,21,24,35,39,40,42,45,46,49,52,56,58,62,65,69,70,71,83,84,87,89,90,96,109,110,116,120,126,128,132,136,138,139,141,146,147,150,156,158,162,166,167,168,170,172,178,181,183,186,193,199,200,201,202,207];
infected_models_ner=[7,16,51,61,86,91,100,104,107,111,121,123,143,159,171,179,192,196];
infected_models_sc=[13,14,34,36,66,72,74,77,99,108,112,113,117,125,134,145,151,154,169,182,187,190,198,205];

import nltk
scores=torch.load('data_r9fuzzd_1_2gram_l10_1x4.pt')
scores=torch.load('data_r9fuzzd_1gram_l20_200k_2x8.pt')
bleu_scores=[];
better=[];
gt=[]
for id in infected_models_sc:
    interface=new(id,nclean=4);
    trigger_gt=interface.trigger_gt;
    score_gt=float(interface.score([interface.fuzzer.encode(trigger_gt)],[5,25]).mean());
    
    tokens=scores['table_ann']['token'][id]
    outputs=scores['table_ann']['score'][id]
    tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
    a,b=tmp.sort(dim=0,descending=True);
    
    best_score=float(a[0])
    best_trigger=interface.fuzzer.decode(tokens[b[0]])
    score_best_trigger=float(interface.score([interface.fuzzer.encode(best_trigger)],[5,25]).mean());
    
    bleu=nltk.translate.bleu_score.sentence_bleu([trigger_gt],best_trigger)
    bleu_scores.append(bleu);
    better.append(score_best_trigger>score_gt)
    gt.append(trigger_gt)

for i in range(20):
    print(a[i],interface.fuzzer.decode(tokens[b[i]]))

bleu_scores
better


outputs=interface.run(tokens,[5,25]);
outputs=outputs.view(len(tokens),-1);
tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
a,b=tmp.sort(dim=0,descending=True);
for i in range(20):
    print(a[i],interface.fuzzer.decode(tokens[b[i]]))




bleu_scores_task=[];
better_task=[];
for id in infected_models_sc:
    i=infected_models.index(id);
    bleu_scores_task.append(bleu_scores[i])
    better_task.append(better[i])



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
scores=torch.load('data_r9fuzzd_2gram_l20_1m_1x4.pt');

id=36
interface=new(id,nclean=4);
tokens=scores['table_ann']['token'][id]
outputs=scores['table_ann']['score'][id]

tmp=(outputs-outputs[0:1,:]).abs().mean(dim=1)
a,b=tmp.sort(dim=0,descending=True);

for i in range(20):
    print(a[i],interface.fuzzer.decode(tokens[b[i]]))

interface.trigger_gt



interface.score([interface.fuzzer.encode(interface.trigger_gt)],[5,25])

interface.score([interface.fuzzer.encode('experienced gender skin diseases began ; station - Joyce and pays. release by the unfavorable draw some replacement cost')],[5,25])


#Get word embeddings

#Load initial word embeddings
from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta=RobertaForMaskedLM.from_pretrained('roberta-base');
we=roberta.roberta.embeddings.word_embeddings.weight.data.float().clone();


'''


