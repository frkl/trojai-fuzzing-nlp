
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


#Provides
#  Inference interface for QA
#  Fuzzing interface for QA

#Load datasets into batches
#Iterate over batches
#Synthesize triggered batches



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

#Dataloading utility
def tokenize_for_qa(tokenizer, dataset):
    from transformers import RobertaTokenizerFast
    if isinstance(tokenizer,RobertaTokenizerFast):
        try:
            tokenizer=torch.load('roberta_tokenizer_fast.pt');
        except:
            try:
                tokenizer=torch.load('/roberta_tokenizer_fast.pt');
            except:
                tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base',add_prefix_space=True);
                torch.save(tokenizer,'roberta_tokenizer_fast.pt')
    
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    #['id', 'tokens', 'ner_tags', 'ner_labels', 'poisoned', 'spurious', 'f1']
    
    
    # Padding side determines if we do (question|context) or (context|question).
    max_seq_length = min(tokenizer.model_max_length, 384)
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples): 
        tokens=tokenizer(examples['tokens'],padding="max_length",truncation=True,is_split_into_words=True,max_length=max_seq_length)
        
        
        token_labels = []
        label_mask = []
        
        word_ids = tokens.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is not None:
                cur_label = examples['ner_tags'][word_idx]
            if word_idx is None:
                token_labels.append(0)
                label_mask.append(0)
            elif word_idx != previous_word_idx:
                token_labels.append(cur_label)
                label_mask.append(1)
            else:
                token_labels.append(0)
                label_mask.append(0)
            previous_word_idx = word_idx
        
        
        tokens['token_labels']=token_labels
        tokens['label_mask']=label_mask
        
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
        #self.model.half();
        self.tokenizer=torch.load(tokenizer_filepath)
    
    def load_examples(self,examples_filepath,scratch_dirpath,bsz=12,shuffle=False):
        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
        
        tokenized_dataset = tokenize_for_qa(self.tokenizer, dataset)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_labels', 'label_mask'])
        
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
                result = self.model(batch['input_ids'],batch['attention_mask'],return_dict=True);
                
                logits=F.log_softmax(result['logits'],dim=2);
                avg_logits=[];
                for i in range(logits.shape[2]):
                    mask=batch['token_labels'].eq(i) & batch['label_mask'].eq(1);
                    avg_logit_i=(logits[:,:,i]*mask).sum(dim=1)/(mask.float().sum(dim=1)+1e-8);
                    avg_logits.append(avg_logit_i);
                
                avg_logits=torch.stack(avg_logits,dim=1);
                outputs.append(avg_logits);
        
        outputs=torch.cat(outputs,dim=0);
        return outputs
    
    


















