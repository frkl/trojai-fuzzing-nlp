import os
import torch
import json
import pandas
import importlib
import math
import jsonschema
import jsonpickle
import time
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore")

import round9_helper as helper

def batch_hist(x,nbins,dim=0):
    assert len(x.shape)==2;
    
    #make histogram dimension dim 1
    if dim==0:
        x=x.t();
    
    h=[];
    for i in range(x.shape[0]):
        v=x[i];
        #v=(v-v.mean()).abs();
        h_i=torch.histc(v,nbins);
        h_i=h_i/h_i.sum();
        h_i=torch.cumsum(h_i,dim=0)
        h.append(h_i);
    
    h=torch.stack(h,dim=0);
    if dim==0:
        h=h.t();
    
    return h

def generate_quantile(N):
    q=((1+10/(N//2-1))**(-torch.arange(N//2-1))).tolist()+[0];
    q2=q[::-1]
    q=torch.Tensor(q)
    q2=torch.Tensor(q2)
    q=torch.cat((q2/2,(1-q)/2+0.5),dim=0)
    return q;


def example_trojan_detector(model_filepath,
                            tokenizer_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            features_filepath,config):
    
    t0=time.time();
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    import fuzzer_nlp_d as fuzzer_nlp
    x,y=fuzzer_nlp.extract_fv(model_filepath, tokenizer_filepath, scratch_dirpath, examples_dirpath,params=config)
    
    print('Preprocessing features, time %.2f'%(time.time()-t0))
    N=300;
    y=y.cuda().float();
    v1=torch.quantile(y,generate_quantile(N).cuda(),dim=0)
    v2=batch_hist(y,N,dim=0);
    y=torch.cat((v1.data,v2.data),dim=0).cpu()
    
    fvs={'token':[x],'score':[y]};
    
    if features_filepath is not None:
        df=pandas.DataFrame(y.tolist());
        df.to_csv(features_filepath);
        print("Features saved to %s"%features_filepath)
    
    #Load config
    import os
    import json
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    
    #task=config['task_type'];
    #if task=='qa':
    #    parameters_dirpath=os.path.join(parameters_dirpath,'qa')
    #elif task=='ner':
    #    parameters_dirpath=os.path.join(parameters_dirpath,'ner')
    #elif task=='sc':
    #    parameters_dirpath=os.path.join(parameters_dirpath,'sc')
    #else:
    #    a=0/0;
    
    if not parameters_dirpath is None:
        checkpoint=os.path.join(parameters_dirpath,'model.pt')
        try:
            checkpoint=torch.load(os.path.join(parameters_dirpath,'model.pt'));
        except:
            checkpoint=torch.load(os.path.join('/',parameters_dirpath,'model.pt'));
        
        #Compute ensemble score 
        scores=[];
        for i in range(len(checkpoint)):
            params_=checkpoint[i]['params'];
            arch_=importlib.import_module(params_.arch);
            net=arch_.new(params_);
            
            net.load_state_dict(checkpoint[i]['net']);
            net=net.cuda();
            net.eval();
            
            s_i=net.logp(fvs).data.cpu();
            s_i=s_i#*math.exp(-checkpoint[i]['T']);
            scores.append(float(s_i))
        
        scores=sum(scores)/len(scores);
        scores=torch.sigmoid(torch.Tensor([scores])); #score -> probability
        trojan_probability=float(scores);
    else:
        trojan_probability=0.5;
    
    print('Trojan Probability: %f, time %.2f'%(trojan_probability,time.time()-t0))
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
    
    return trojan_probability




def configure(output_parameters_dirpath,
              configure_models_dirpath,
              parameter3=None):
    print('Using parameter3 = {}'.format(str(parameter3)))

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    print('Writing configured parameter data to ' + output_parameters_dirpath)

    arr = np.random.rand(100,100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(example_dict, warn=True, indent=2))



if __name__ == "__main__":
    #import argparse
    #args ,b = argparse.ArgumentParser().parse_known_args()
    #print(args,b)
    t0=time.time();
    from jsonargparse import ArgumentParser, ActionConfigFile
    
    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')
    
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)
    
    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')
    
    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')
    
    
    
    parser.add_argument('--nclean', type=int, help='Number of clean examples to use.')
    parser.add_argument('--bsz', type=int, help='Trigger search batch size')
    parser.add_argument('--fuzzer_checkpoint', type=str, help='The search schedule of the trigger search module')
    
    args= parser.parse_args()
    
    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)
            
            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)
            
            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)
    
    
    
    
    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.round_training_dataset_dirpath is not None and
                args.learned_parameters_dirpath is not None):

            example_trojan_detector(args.model_filepath,
                                    args.tokenizer_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.features_filepath,
                                    args,
                                    )
        else:
            print("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None):

            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath)
        else:
            print("Required Configure-Mode parameters missing!")
    
    print('Total time %f'%(time.time()-t0))