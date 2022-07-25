
import os
import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
import json
import jsonschema
import jsonpickle
import warnings
import copy
import torch.nn.functional as F


#Provides
#  Inference interface for QA
#  Fuzzing interface for QA

#Load datasets into batches
#Iterate over batches
#Synthesize triggered batches



#Dataloading utility
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    #['id', 'tokens', 'ner_tags', 'ner_labels', 'poisoned', 'spurious', 'f1']
    
    
    # Padding side determines if we do (question|context) or (context|question).
    max_seq_length = min(tokenizer.model_max_length, 150)
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        tokens=tokenizer(examples['data'],padding="max_length",truncation=True,max_length=max_seq_length)
        tokens['label']=examples['label']
        return tokens
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=False,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_labels': [],
                     'label_mask': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset

def mask_logsoftmax(score,mask,dim=1):
    score=score-(1-mask)*1e5;
    return F.log_softmax(score,dim=dim);

class new:
    def __init__(self,model_filepath, tokenizer_filepath):
        self.model=torch.load(model_filepath).cuda();
        self.model.half();
        self.model.eval()
        self.model=nn.DataParallel(self.model);
        self.tokenizer=torch.load(tokenizer_filepath)
        self.max_seq_length = self.tokenizer.model_max_length
    
    def load_examples(self,examples_filepath,scratch_dirpath,bsz=12,shuffle=False):
        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
        
        tokenized_dataset = tokenize_for_qa(self.tokenizer, dataset)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'label'])
        
        dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=bsz,shuffle=shuffle);
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
                
                
                #Trim unused tokens & sequences too long
                attention_mask=batch['attention_mask']
                input_ids=batch['input_ids']
                label=batch['label']
                
                ind=attention_mask[:,:self.max_seq_length].sum(dim=0).nonzero().view(-1); #all tokens in use
                #print(len(ind))
                input_ids=input_ids[:,ind].contiguous();
                attention_mask=attention_mask[:,ind].contiguous();
                batch={'input_ids':input_ids,'attention_mask':attention_mask,'label':label}
                
                
                
                
                result = self.model(batch['input_ids'],batch['attention_mask'],return_dict=True);
                logits=F.log_softmax(result['logits'],dim=1);
                #order logits by gt vs nongt
                logits_gt=logits.gather(1,batch['label'].view(-1,1));
                logits_nongt=logits.gather(1,1-batch['label'].view(-1,1));
                logits=torch.cat((logits,logits_gt,logits_nongt),dim=1);
                
                outputs.append(logits);
        
        outputs=torch.cat(outputs,dim=0);
        return outputs
    
    


















