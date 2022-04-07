
import os
import datasets
import numpy as np
import torch
import transformers
import json
import jsonschema
import jsonpickle
import warnings


warnings.filterwarnings("ignore")

def get_paths(id,root='data/round9-train-dataset'):
    id='id-%08d'%id;
    f=open(os.path.join(root,'models',id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,'models',id,'model.pt');
    examples_dirpath=os.path.join(root,'models',id,'clean-example-data.json');
    scratch_dirpath='./scratch'
    if 'electra' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/google-electra-small-discriminator.pt');
    elif 'distilbert' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/distilbert-base-cased.pt');
    elif 'roberta' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/roberta-base.pt');
    else:
        a=0/0;
    
    return model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath;

def get_paths_r8(id,root='data/round8-train-dataset'):
    id='id-%08d'%id;
    f=open(os.path.join(root,'models',id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,'models',id,'model.pt');
    examples_dirpath=os.path.join(root,'models',id,'example_data','clean-example-data.json');
    scratch_dirpath='./scratch'
    if 'electra' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/tokenizer-google-electra-small-discriminator.pt');
    elif 'deepset' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/tokenizer-deepset-roberta-base-squad2.pt');
    elif 'roberta' in config['model_architecture']:
        tokenizer_filepath=os.path.join(root,'tokenizers/tokenizer-roberta-base.pt');
    else:
        a=0/0;
    
    return model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath;

def get_paths_r7(id,root='data/round7-train-dataset'):
    id='id-%08d'%id;
    f=open(os.path.join(root,'models',id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,'models',id,'model.pt');
    examples_dirpath=os.path.join(root,'models',id,'clean_example_data');
    scratch_dirpath='./scratch'
    if 'MobileBERT' in config['embedding']:
        tokenizer_filepath=os.path.join(root,'tokenizers/MobileBERT-google-mobilebert-uncased.pt');
    elif 'RoBERTa' in config['embedding']:
        tokenizer_filepath=os.path.join(root,'tokenizers/RoBERTa-roberta-base.pt');
    elif 'DistilBERT' in config['embedding']:
        tokenizer_filepath=os.path.join(root,'tokenizers/DistilBERT-distilbert-base-cased.pt');
    elif 'BERT' in config['embedding']:
        tokenizer_filepath=os.path.join(root,'tokenizers/BERT-bert-base-uncased.pt');
    else:
        a=0/0;
    
    return model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath;


def get_paths_r6(id,root='data/round6-train-dataset'): # Model loading doesn't work due to pytorch version difference 
    id='id-%08d'%id;
    f=open(os.path.join(root,'models',id,'config.json'),'r');
    config=json.load(f);
    f.close();
    
    model_filepath=os.path.join(root,'models',id,'model.pt');
    examples_dirpath=os.path.join(root,'models',id,'clean_example_data');
    scratch_dirpath='./scratch'
    if 'distilbert-base-uncased' in config['embedding_flavor']:
        tokenizer_filepath=os.path.join(root,'tokenizers/DistilBERT-distilbert-base-uncased.pt');
    elif 'gpt2' in config['embedding_flavor']:
        tokenizer_filepath=os.path.join(root,'tokenizers/GPT-2-gpt2.pt');
    else:
        a=0/0;
    
    return model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath;


def load_stuff(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath):
    # load the classification model and move it to the GPU
    pytorch_model = torch.load(model_filepath).cuda()
    pytorch_model.eval()
    tokenizer = torch.load(tokenizer_filepath)
    
    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #Load example data
    if examples_dirpath.endswith('.json'):
        fns=[examples_dirpath];
    else:
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    
    fns.sort()
    examples_filepath = fns[0]
    
    #Create dataset based on task
    if config['task_type']=='qa':
        pass;
    elif  config['task_type']=='sc':
        pass;
    elif  config['task_type']=='ner':
        pass;
    else:
        print('Unrecognized task type: %s'%config['task_type']);
    
    
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
    
    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    
    # Load the provided tokenizer
    # TODO: Use this method to load tokenizer on T&E server
    
    # TODO: This should only be used to test on personal machines, and should be commented out
    #  before submitting to evaluation server, use above method when submitting to T&E servers
    # model_architecture = config['model_architecture']
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)
    
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
    
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)
    
    return pytorch_model,tokenizer,dataset,tokenized_dataset,dataloader


# The inference approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
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


