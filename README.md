This repo implements a minimal fuzzing-based framework for Trojan detection for NLP tasks. 

## Data

Download TrojAI round 9 models and extract models as `trojai-fuzzing-nlp/data/round9-train-dataset/models/id-00000xxx`, tokenizers as `trojai-fuzzing-nlp/data/round9-train-dataset/tokenizers/roberta-base.pt`, `google-electra-small-discriminator.pt` and `distilbert-base-cased.pt` respectively.

## Usage

Clone this repository. Setup environment following https://github.com/usnistgov/trojai-example .

Download pretrained Trojan classifier checkpoint at https://www.dropbox.com/s/zzkdfd7gz3jn4ih/model.pt?dl=1 and put it under `trojai-fuzzing-nlp/learned_parameters/`.

Run `build.sh` to evaluate the fuzzer on model `id-00000108` (assuming you have TrojAI round 9 data downloaded, see data section). You should see ~90% Trojan probability.

`build.sh` also builds a container with `trojan_detector.def`. Building the container does not require downloading TrojAI data.


## System architecture

Our system has 3 components

-- Entry point `trojan_detector.py`. Calls fuzzing library `fuzzer.py` to extract features on the model, and then runs the pretrained trojan detector in `learned_parameters` for trojan classification

-- Fuzzing library `fuzzer_nlp_c.py`, which loads the model and clean examples, and insert random triggers to the clean examples and inference the model. Model outputs on multiple clean examples and at multiple trigger insertion locations are used as features to capture how much the outputs are affected by the random triggers for Trojan detection. Developing smarter ways to generate triggers is a promising direction of research.

-- Fuzzing interfaces for each NLP task `ner_engine_v1c.py`, `qa_engine_v1c.py` and `sc_engine_v1c.py` which defines how clean examples are loaded, how triggers are inserted, how to inference the model and which outputs should be used. Specifically, each fuzzing interface implements a `load_examples` method which takes example paths and returns a set of examples, a `insert_trigger` method which takes a trigger and a set of examples and returns a triggered set of examples, and a `inferece` method which takes a set of examples and return their scores. How triggers are inserted and which scores should be recorded may affect Trojan detection performance and is a promising direction of research.

## Learning

To generate a pretrained trojan detector, we'll need to first run `python fuzzer.py` to extract features on existing models into `data_r9fuzzc_e2s.pt`. This process may take a few hours.

Once feature extraction finishes, run `python crossval_hyper_s3.py --data data_r9fuzzc_e2s.pt --arch arch.mlp_known_v7s3N ` to train a Trojan detector on random triggers. It is a hyperparam search process using cross validation. At any time, the best model so far is saved in `trojai-fuzzing-nlp/sessions/xxxxxxx`. Manually copy it out as the `learned_parameters` folder for submission.


## Known issues

We have not yet implemented the `configure` call in our container. It is currently a placeholder.
