This repo implements a minimal fuzzing-based framework for Trojan detection for NLP tasks. 

## Usage

Clone this repository. Download TrojAI round 9 models and extract models as `trojai-fuzzing-nlp/data/round9-train-dataset/models/id-00000xxx`, tokenizers as `trojai-fuzzing-nlp/data/round9-train-dataset/tokenizers/roberta-base.pt`, `google-electra-small-discriminator.pt` and `distilbert-base-cased.pt` respectively.

Run `build.sh` to evaluate the fuzzer on model `id-00000000`. You should see

`build.sh` also builds a container with `trojan_detector.def`.

## System architecture

Our system has 3 components

-- Entry point `trojan_detector.py`. Calls fuzzing library `fuzzer.py` to extract features on the model, and then runs the pretrained trojan detector in `learned_parameters` for trojan classification

-- Fuzzing library `fuzzer.py`, which loads the model and clean examples, and insert random triggers to the clean examples and inference the model. Model outputs are used as features to capture how much the outputs are affected by the random triggers for Trojan detection. Smarter ways to generate triggers is a promising direction for research.

-- Fuzzing interfaces for each NLP task `ner_engine.py`, `qa_engine.py` and `sc_engine.py` which defines how clean examples are loaded, how triggers are inserted, how to inference the model and which outputs should be used. Specifically, each fuzzing interface implements a `load_examples` method which takes example paths and returns a set of examples, a `insert_trigger` method which takes a trigger and a set of examples and returns a triggered set of examples, and a `inferece` method which takes a set of examples and return their scores. How triggers are inserted and which scores should be recorded may affect Trojan detection performance and is a promising direction for research.

