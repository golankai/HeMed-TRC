
# HeMed: A Hebrew Medical Language Model for Clinical Timeline Extraction

This repository contains the code and infrastructure for **HeMed**, a Hebrew language model trained on real-life clinical data. It also provides an anonymiszation pipeline and an evaluation code for the temporal relation classification (TRC) task. The code, model and results are introduced in the paper:

**_Building Patient Journeys in Hebrew: A Language Model for Clinical Timeline Extraction_**,  
*Large Language Models and Generative AI for Health Informatics Workshop at IJCAI 2025*.

---

## Overview

**HeMed** is designed for medical text mining in Hebrew. It supports:

- Domain adaptation of existing PLMs
- Continual pretraining on (clinical) data
- De-identification (anonymization)
- Evaluation on (clinical) TRC tasks

---

## Project Structure

All code is in the `project/` directory.

| Folder/File               | Description |
|--------------------------|-------------|
| `domain_adaptation/`     | Tokenizer adaptation (simple extension, AdaLM). |
| `further_pre_training/`  | Continual pretraining pipeline for base PLMs. |
| `evaluation/`            | Model evaluation on TRC NLP. |
| `safe_harbor/`           | De-identification with regex or [HebSafeHarbor](https://github.com/8400TheHealthNetwork/HebSafeHarbor). |
| `anonymize.py`           | Script to run the de-identification process. |
| `calc_ctc.py`            | Script to calculate tokenizer metrics of a given corpus. |
| `run_exp.py`             | Main runner script controlled via `config.json`. |
| `utils.py`               | Shared utilities, base model names, helpers. |

---

## ⚙️ Setup

### Create Environment

```bash
conda create -n hemed python=3.11
conda activate hemed
pip install -r requirements.txt
cd project
```
> **CUDA 12.1** is required for GPU-based training. Set the GPU index at the top of `run_exp.py`.

> For **de-identification**, use Python 3.8.19 and install requirements from `safe_harbor/requirements.txt`.


---

## Running an Experiment

All components are configured via `config.json` and the sub-configs in their respective folders. The system is modular — you can run domain adaptation, continual pretraining, evaluation, or all together in a single pipeline.

To run an experiment:
```bash
python run_exp.py
```

---

## Domain Adaptation

To adapt a tokenizer:

1. In `config.json`, set: `"do_da": true`
2. Edit `domain_adaptation/config.json`:
   - Specify `tokenizer_base_names`
   - Select `schemas` and related parameters

We support both **simple tokenizer extension** and **AdaLM adaptation**.

---

## Continual Pretraining

To further pretrain a PLM:

1. In `config.json`, set: `"do_pretraining": true`
2. In `further_pre_training/config.json`:
   - Set `train_file` (txt, json, csv, parquet)
   - Choose `model_base_name` from:

```python
{
    "hebert": "avichr/heBERT",
    "alephbert": "onlplab/alephbert-base",
    "abg": "imvladikon/alephbertgimmel-base-512",
    "hero": "HeNLP/HeRo",
    "dictabert": "dicta-il/dictabert"
}
```

---

## Evaluation & Fine-Tuning

To evaluate/fine-tune:

1. In `config.json`, set: `"eval": true`
2. In `evaluation/config.json`, configure:

**Options:**
- Fine-tune one specific model
- Fine-tune **all** locally available models (`"evaluate_local_models": true`)
- Evaluate base PLMs from Hugging Face (`"evaluate_hf_models": true`, specify in `"hf_models_to_train"`)

> ⚠️ Note: If `evaluate_local_models` is true, set `do_pretraining` to false in the general config.

---

## 🔐 De-identification

Run `anonymize.py` for de-identifying medical text via regex-based rules or use the [HebSafeHarbor](https://github.com/8400TheHealthNetwork/HebSafeHarbor) tool. In this script are some configurations to set.

---

## Citation

If you use this code or reference HeMed in your work, please cite:

```bibtex
@inproceedings{hashiloni2025hemed,
  title     = {Building Patient Journeys in Hebrew: A Language Model for Clinical Timeline Extraction},
  author    = {Golan Hashiloni, Kai and others},
  booktitle = {Proceedings of the LLM4Health Workshop, IJCAI},
  year      = {2025}
}
```

---

## 📬 Contact

For questions or contributions, open an issue or contact the authors directly.
