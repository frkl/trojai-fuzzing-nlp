
import os
import datasets
import numpy as np
import torch
import transformers
import json
import jsonschema
import jsonpickle
import warnings
import copy
import torch.nn.functional as F


class dataloader_r6:
    def __init__(self,root,tokenizer):
        text_fnames=[os.path.join(examples_dirpath,f) for f in os.listdir(examples_dirpath) if f.endswith('.txt')];
        text_fnames=sorted(text_fnames);
        data=[];
        labels=[];
        for fname in text_fnames:
            with open(fname,'r') as f:
                text=f.read();
            
            data.append(text);
            labels.append(int(fname.split('_')[-3]));
        
        self.data=data;
        self.labels=labels;
        
        self.tokenizer=tokenizer
        # Padding side determines if we do (question|context) or (context|question).
        max_seq_length = tokenizer.model_max_length
        if 'mobilebert' in tokenizer.name_or_path:
            max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
        
        max_seq_length=min(max_seq_length,384);
        self.max_input_length=max_seq_length
        return
    
    def select_by_index(self,ind):
        batch=[];
        for i in ind:
            batch.append(self[i]);
        return batch;
    
    def __getitem__(self,index):
        results=self.tokenizer(self.data[i], max_length=self.max_input_length,padding=True,truncation=True,return_tensors="pt")
        input_ids=results.data['input_ids']
        attention_mask=results.data['attention_mask']
        return {'input_ids':input_ids,'attention_mask':attention_mask,'label':self.labels[i]};
    
    def __len__(self):
        return len(self.clean_fnames)


#Combines multiple sentences into one
def collate_fn(batch,pad=0):
    token=[x['input_ids'] for x in batch]
    token_mask=[x['attention_mask'] for x in batch]
    label=[x['label'] for x in batch]
    
    maxl=max([len(x) for x in token]);
    assert(maxl>=max([len(x) for x in token_mask]))
    
    #Pad everything to maxl
    token=[F.pad(x,(0,maxl-len(x)),value=pad) for x in token]; #assming [PAD]=0
    token_mask=[F.pad(x,(0,maxl-len(x)),value=0) for x in token_mask];
    
    return {'input_ids':torch.stack(token,dim=0),'attention_mask':torch.stack(token_mask,dim=0),'label':torch.stack(label,dim=0)}


def mask_logsoftmax(score,mask,dim=1):
    score=score-(1-mask)*1e5;
    return F.log_softmax(score,dim=dim);

class new:
    def __init__(self,model_filepath, tokenizer_filepath):
        self.model=torch.load(model_filepath).cuda();
        self.model.half();
        self.tokenizer=torch.load(tokenizer_filepath)
    
    def load_examples(self,examples_filepath,scratch_dirpath,bsz=12,shuffle=False):
        dataset=dataloader_r6(examples_filepath,self.tokenizer);
        dataloader=torch.utils.data.DataLoader(dataset,bsz,collate_fn=collate_fn,shuffle=shuffle,num_workers=0)
        examples=[x for x in dataloader];
        return examples;
    
    def insert_trigger(self,examples,trigger,start_idx):
        trigger=self.tokenizer(trigger)['input_ids'][1:-1];
        trigger=trigger[:20]; #Cap trigger size at 20
        trigger_length=len(trigger);
        triggered_examples=[];
        
        for d in examples:
            t=torch.LongTensor(trigger).clone().view(1,-1);
            t=t.repeat(d['input_ids'].shape[0],1);
            
            min_length=d['attention_mask'].sum(dim=-1).min();
            start_idx_i=min(start_idx,min_length);
            
            input_ids=torch.cat((d['input_ids'][:,:start_idx_i],t,d['input_ids'][:,start_idx_i:]),dim=1);
            attention_mask=torch.cat((d['attention_mask'][:,:start_idx_i],t*0+1,d['attention_mask'][:,start_idx_i:]),dim=1);
            
            d_triggered={};
            d_triggered['input_ids']=input_ids;
            d_triggered['attention_mask']=attention_mask;
            d_triggered['label']=d['label'];
            
            triggered_examples.append(d_triggered);
        
        return triggered_examples;
    
    #Perform inference
    #Compute 
    #1. loss at GT
    #2. loss at token 0
    #3. See if we can compute some hidden feature dims
    def inference(self,examples):
        with torch.no_grad():
            outputs=[];
            for batch in examples:
                batch=dict([(k,batch[k].cuda()) for k in batch]);
                result = self.model(batch['input_ids'],batch['attention_mask'],return_dict=True);
                logits=F.log_softmax(result['logits'],dim=1);
                outputs.append(logits);
        
        outputs=torch.cat(outputs,dim=0);
        return outputs
    
    


















