
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

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)

    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    max_seq_length = min(max_seq_length-20, 360) #384->360 just in case of overflow
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)

    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
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
        self.model=nn.DataParallel(self.model)
        self.tokenizer=torch.load(tokenizer_filepath)
    
    def load_examples(self,examples_filepath,scratch_dirpath,bsz=12,shuffle=False):
        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
        
        tokenized_dataset = tokenize_for_qa(self.tokenizer, dataset)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
        
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
            token_type_ids=torch.cat((d['token_type_ids'][:,:start_idx_i],t*0,d['token_type_ids'][:,start_idx_i:]),dim=1);
            start_positions=copy.deepcopy(d['start_positions']);
            end_positions=copy.deepcopy(d['end_positions']);
            start_positions[start_positions.gt(start_idx_i)]+=trigger_length;
            end_positions[end_positions.gt(start_idx_i)]+=trigger_length;
            
            d_triggered={};
            d_triggered['input_ids']=input_ids;
            d_triggered['attention_mask']=attention_mask;
            d_triggered['token_type_ids']=token_type_ids;
            d_triggered['start_positions']=start_positions;
            d_triggered['end_positions']=end_positions;
            
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
                try:
                    result = self.model(**batch);
                except:
                    result = self.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']);
                start_logits=result['start_logits']
                end_logits=result['end_logits'];
                
                start_logits=mask_logsoftmax(start_logits,batch['attention_mask'],dim=1);
                end_logits=mask_logsoftmax(end_logits,batch['attention_mask'],dim=1);
                start_loss=-start_logits.gather(1,batch['start_positions'].view(-1,1));
                end_loss=-end_logits.gather(1,batch['end_positions'].view(-1,1));
                
                loss=(start_loss+end_loss).view(-1)/2;
                loss_0=-(start_logits[:,0]+end_logits[:,0])/2;
                
                output=torch.cat((loss.view(-1,1),loss_0.view(-1,1),start_loss.view(-1,1),end_loss.view(-1,1),start_logits[:,0:1],end_logits[:,0:1]),dim=1); #batch x 6
                #output=torch.cat((loss.view(-1,1),loss_0.view(-1,1)),dim=1); #batch x 2
                outputs.append(output);
        
        outputs=torch.cat(outputs,dim=0);
        return outputs
    
    


















