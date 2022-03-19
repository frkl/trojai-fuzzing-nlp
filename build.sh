python trojan_detector.py  --model_filepath=./data/round9-train-dataset/models/id-00000000/model.pt  --tokenizer_filepath=./data/round9-train-dataset/tokenizers/google-electra-small-discriminator.pt  --features_filepath=./features.csv  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=./data/round9-train-dataset/models/id-00000000/clean-example-data.json  --round_training_dataset_dirpath=/path/to/training/dataset/  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json  --learned_parameters_dirpath=./learned_parameters
 
 
 
sudo singularity build test-trojai-r9-v3.simg trojan_detector.def 
