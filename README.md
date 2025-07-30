# HeMed: A Medical Language Model Trained on Real Life Clinical Data
This repository contains the code for pre-training and fine-tuning a Hebrew LM specialized in medical-clinic Language - HeMed.

## Project Overview
This repository contains the code for training and evaluating a Hebrew LM specialized in medical-clinic Language - HeMed. More specifically, we supply here the infrastructure to perform continual pre-training of a base LM on medical data and then evaluate its performance on downstream tasks.

## Project Structure
The project's code is in the `project` folder.
1. `domain_adaptation` contains the code to perform tokenizer adaptation of an existing base tokenizer (model) to a domain specific corpus. Options are simple tokenizer extension or the AdaLM method (Yao et al., [2021](
https://doi.org/10.48550/arXiv.2106.13474)).
1. `further_pre_training` contains the code to do continual pre-trainin of a base model on your own data. It loads a base PLM and further pretrains it using the (modified) [`run_mlm.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) script from Hugging Face (Hf).
1. `evaluation` contains the evaluation process of a model. You can load again a PLM from HF or load a local model. The model will then be evaluated on the desired tasks, can be adapted.
1. `safe_harbor` contains code to support the `anonymize.py` script which performs de-identification of a given text corpus. One can adapt the regex-based method we applied ot use the out-of-the-box [`SafeHarbor`](https://github.com/8400TheHealthNetwork/HebSafeHarbor) tool.
1. `summary` contains the experiment's logs that our produces by the `run_exp.py` script.
1. `translation` contains infrastructure we have used to auto-tranlated a dataset. One can easily adapt it to any languages.

## Usage
### Set up the Environment
Create a conda environment:
```
conda create -n hemed python=3.11.0
conda activate hemed
pip install -r requirement.txt
cd project
```
Not that for the de-identification process one needs a different environment with `python==3.8.19` and and the requirements detailts in `anonymization/requirements.txt`.


If you work with a GPU, please indicate its number at the top of the `run_exp.py` file. This will indicate some libraries with which GPU to work with. This project requires `CUDA==12.1`.


### Set up an Experiment
First, go to the `project` folder, where you design and run the experiment.
The Experiment is designed using configuration files. The main one, indicating which parts to run, seed, etc. is the `config.json` file under the `project` folder. All steps can be run continuously one after the other, if you wish.

Based on your Experiment, edit the respective `config.json` file of the subdirectories and run this command:
```
python run_exp.py
```

### Domain Adaptation
First under the *general* config file, make sure `"do_da": true`.
Then under the *domain adaptation* config file, adapt your parameters and configuration as you wish. Under `tokenizer_base_names` add the names of tokenizer (models) to adapt, see below the options available. Note that for each schema under `schemas`, you should adapt the relevant parameter below in the same file.

### Continual Pre-Train a model
First under the *general* config file, make sure `"do_pretraining": true`.
Then under the *further pretraining* config file, adapt your parameters and configuration as you wish. Most important are the `train_file` and the `model_base_name` entries, where the first should be a [txt, csv, json, parquet] file with your training data and the second a base model name from those keys (can be adapted in the `utils.py` file) :
```
{
    "hebert": "avichr/heBERT",
    "alephbert": "onlplab/alephbert-base",
    "abg": "imvladikon/alephbertgimmel-base-512",
    "hero": "HeNLP/HeRo",
    "dictabert": "dicta-il/dictabert",
}
```
        
### Fine-tunning a model
First under the *general* config file, make sure `"eval": true`.
Then under the *evaluation* config file, adapt your parameters and configuration as you wish. Usage options:
1. Fine-tunning a further-pretrained model directly.
1. Fine-tunning all existing further pretrained models - in the trc config file, make sure `"evaluate_local_models": true`. **Important** - in the *general* config file `do_pretraining` must be false! You can't further train a model and evaluate all **existing ones** at once.
1. Fine-tunning a base PLM - if you want to run only the evaluation pipeline, indicate `"evaluate_local_models": false`, `"evaluate_hf_models": true` and specify which base models to use in `"hf_models_to_train": []`. The names as the keys of the dictionary above.

## Contact
If you have any quaries or comments, please open an issue or contact us privately!

## Cite
If using the code or mentioning our paper or results, please cite as:
```
{
    ADD
}
```
