
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
import torch.utils.data


#Provides
#  Inference interface for QA
#  Fuzzing interface for QA

#Load datasets into batches
#Iterate over batches
#Synthesize triggered batches


class dataloader_r7:
    def __init__(self,root,tokenizer):
        clean_fnames = [f for f in os.listdir(root) if f.endswith('.txt') and not f.endswith('_tokenized.txt')]
        clean_fnames.sort()
        
        self.clean_fnames=[os.path.join(root,fname) for fname in clean_fnames]
        
        self.tokenizer=tokenizer
        # Padding side determines if we do (question|context) or (context|question).
        max_seq_length = tokenizer.model_max_length
        if 'mobilebert' in tokenizer.name_or_path:
            max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
        
        max_seq_length=min(max_seq_length,384);
        self.max_input_length=max_seq_length
        return
    
    def tokenize_and_align_labels(self,words,labels):
        tokens=self.tokenizer(words,padding=True,truncation=True,is_split_into_words=True,max_length=self.max_input_length)
        
        token_labels = []
        label_mask = []
        
        word_ids = tokens.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is not None:
                cur_label = labels[word_idx]
            if word_idx is None:
                token_labels.append(-100)
                label_mask.append(0)
            elif word_idx != previous_word_idx:
                token_labels.append(cur_label)
                label_mask.append(1)
            else:
                token_labels.append(-100)
                label_mask.append(0)
            previous_word_idx = word_idx
            
        return tokens['input_ids'], tokens['attention_mask'], token_labels, label_mask

    def select_by_index(self,ind):
        batch=[];
        for i in ind:
            batch.append(self[i]);
        return batch;
    
    def __getitem__(self,index):
        words=[];
        labels=[];
        fname=self.clean_fnames[index];
        with open(fname,'r') as f:
            lines=f.readlines();
            for line in lines:
                data=line.split('\t');
                word=data[0].strip();
                label=data[2].strip();
                words.append(word);
                labels.append(int(label));
        
        input_id,attention_mask,label,mask=self.tokenize_and_align_labels(words,labels);
        input_id=torch.LongTensor(input_id)
        attention_mask=torch.LongTensor(attention_mask)
        label=torch.LongTensor(label)
        mask=torch.LongTensor(mask)
        return {'input_ids':input_id,'attention_mask':attention_mask,'token_labels':label,'label_mask':mask,'data':' '.join(words)};
    
    def __len__(self):
        return len(self.clean_fnames)


#Combines multiple sentences into one
def collate_fn(batch,pad=0):
    token=[x['input_ids'] for x in batch]
    token_mask=[x['attention_mask'] for x in batch]
    label=[x['token_labels'] for x in batch]
    label_mask=[x['label_mask'] for x in batch]
    sents=[x['data'] for x in batch]
    
    maxl=max([len(x) for x in token]);
    assert(maxl>=max([len(x) for x in token_mask]))
    assert(maxl>=max([len(x) for x in label]))
    assert(maxl>=max([len(x) for x in label_mask]))
    
    #Pad everything to maxl
    token=[F.pad(x,(0,maxl-len(x)),value=pad) for x in token]; #assming [PAD]=0
    token_mask=[F.pad(x,(0,maxl-len(x)),value=0) for x in token_mask];
    label=[F.pad(x,(0,maxl-len(x)),value=-100) for x in label];
    label_mask=[F.pad(x,(0,maxl-len(x)),value=0) for x in label_mask];
    
    return {'input_ids':torch.stack(token,dim=0),'attention_mask':torch.stack(token_mask,dim=0),'token_labels':torch.stack(label,dim=0),'label_mask':torch.stack(label_mask,dim=0),'data':sents}
    
    

class new:
    def __init__(self,model_filepath, tokenizer_filepath):
        self.model=torch.load(model_filepath).cuda();
        #self.model.half();
        self.tokenizer=torch.load(tokenizer_filepath)
    
    def load_examples(self,examples_filepath,scratch_dirpath,bsz=12,shuffle=False):
        dataset=dataloader_r7(examples_filepath,self.tokenizer);
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
            token_labels=torch.cat((d['token_labels'][:,:start_idx_i],t*0,d['token_labels'][:,start_idx_i:]),dim=1);
            label_mask=torch.cat((d['label_mask'][:,:start_idx_i],t*0,d['label_mask'][:,start_idx_i:]),dim=1);
            
            
            d_triggered={};
            d_triggered['input_ids']=input_ids;
            d_triggered['attention_mask']=attention_mask;
            d_triggered['token_labels']=token_labels;
            d_triggered['label_mask']=label_mask;
            
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
                _,logits = self.model(batch['input_ids'],batch['attention_mask']); 
                logits=F.log_softmax(logits,dim=2);
                avg_logits=[];
                for i in range(logits.shape[2]):
                    mask=batch['token_labels'].eq(i) & batch['label_mask'].eq(1);
                    avg_logit_i=(logits[:,:,i]*mask).sum(dim=1)/(mask.float().sum(dim=1)+1e-8);
                    avg_logits.append(avg_logit_i);
                
                avg_logits=torch.stack(avg_logits,dim=1);
                outputs.append(avg_logits);
        
        outputs=torch.cat(outputs,dim=0);
        return outputs
    
    


















