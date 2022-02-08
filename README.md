This repo implements a minimal fuzzing-based framework for Trojan detection for NLP tasks. 

## Usage

Clone this repository. Download TrojAI round 9 models and extract models as `trojai-fuzzing-nlp/data/round9-train-dataset/models/id-00000xxx`, tokenizers as `trojai-fuzzing-nlp/data/round9-train-dataset/tokenizers/roberta-base.pt`, `google-electra-small-discriminator.pt` and `distilbert-base-cased.pt` respectively.

Run `build.sh` to evaluate the fuzzer on model `id-00000000`. You should see

```
$ ./build.sh
example_trojan_detector device confirmed, time 0.051344
fuzzer importing 0 0.000003
fuzzer importing 1 0.000036
fuzzer importing 2 0.000051
fuzzer importing 3 0.000056
fuzzer importing 4 0.005519
fuzzer importing 5 0.006966
fuzzer importing 6 0.006982
fuzzer importing 7 0.006999
example_trojan_detector Starting feature extraction, time 0.060710
extract_fv_ Task qa, time 0.00
extract_fv_ Loading examples, time 2.48
Using custom data configuration default-8ceb0b7f1be860db
Downloading and preparing dataset json/default to ./scratch/.cache/json/default-8ceb0b7f1be860db/0.0.0/c90812beea906fcffe0d5e3bb9eba909a80a998b5f88e9f8acbd320aa91acfde...
100%|███████████████████████████████████████████| 1/1 [00:00<00:00, 4288.65it/s]
100%|████████████████████████████████████████████| 1/1 [00:00<00:00, 988.52it/s]
Dataset json downloaded and prepared to ./scratch/.cache/json/default-8ceb0b7f1be860db/0.0.0/c90812beea906fcffe0d5e3bb9eba909a80a998b5f88e9f8acbd320aa91acfde. Subsequent calls will reuse this data.
100%|█████████████████████████████████████████████| 1/1 [00:00<00:00, 24.68ba/s]
extract_fv_ Loading fuzzer, time 2.97
extract_fv_ Run fuzzer, time 10.24
torch.Size([102, 8]) torch.Size([102, 126])
extract_fv_ Fuzzing done, time 25.16
example_trojan_detector Feature extracted, time 25.237767
Features saved to ./features.csv
example_trojan_detector Feature saved, time 25.274393
example_trojan_detector Trojan classifier loaded, time 25.308890
example_trojan_detector Trojan classifier calculated. Saving to file, time 25.492256
example_trojan_detector Trojan Probability: 0.221603, time 25.492488
Total time 25.636181

```

`build.sh` also builds a container with `trojan_detector.def`.

## System architecture

Our system has 3 components

-- Entry point `trojan_detector.py`. Calls fuzzing library `fuzzer.py` to extract features on the model, and then runs the pretrained trojan detector in `learned_parameters` for trojan classification

-- Fuzzing library `fuzzer.py`, which loads the model and clean examples, and insert random triggers to the clean examples and inference the model. Model outputs are used as features to capture how much the outputs are affected by the random triggers for Trojan detection. Smarter ways to generate triggers is a promising direction for research.

-- Fuzzing interfaces for each NLP task `ner_engine.py`, `qa_engine.py` and `sc_engine.py` which defines how clean examples are loaded, how triggers are inserted, how to inference the model and which outputs should be used. Specifically, each fuzzing interface implements a `load_examples` method which takes example paths and returns a set of examples, a `insert_trigger` method which takes a trigger and a set of examples and returns a triggered set of examples, and a `inferece` method which takes a set of examples and return their scores. How triggers are inserted and which scores should be recorded may affect Trojan detection performance and is a promising direction for research.
