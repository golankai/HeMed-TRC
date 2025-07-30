"""
A script to calculate the CTC of a bunch of tokenizers on a dataset.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import set_seed, AutoTokenizer  # noqa: E402
import warnings  # noqa: E402
from tqdm import tqdm  # noqa: E402

from further_pre_training.utils import FlotaBERTTokenizer  # noqa: E402

# Ignore watnings about tokenization too long sequences
warnings.filterwarnings("ignore", message=r"Token indices sequence.*")

def calc_metrics(tokenizer_path: str, dataset, flota: bool = False, k: int = 10) -> int:
    # Calculate word count
    word_count = sum(len(x.split()) for x in dataset["text"])

    # Load tokenizer, if Flota, use use non-Fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=not flota)
    
    # If Flota, use FlotaTokenizer
    if flota:
        tokenizer.wordpiece_tokenizer = FlotaBERTTokenizer(tokenizer, k=k)

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=False, add_special_tokens=False),
        batched=True,
        desc="Tokenizing test corpus",
    )

    ctc = sum(len(x) for x in tokenized_dataset["input_ids"])

    # Calculate compression rate
    compression_rate = ctc / word_count
    return ctc, compression_rate


def main():
    # Set configurations
    DEBUG = False
    SEED = 42

    tokenizers_dir = "domain_adaptation/tokenizers/to_eval/"
    data_dir = "further_pre_training/data/cut_ichilov/"
    data_file = "ichilov_test_10k_cut.parquet"
    data_path = data_dir + data_file
    output_file = tokenizers_dir + data_file.split(".")[0] + "_ctc.csv"

    # Delete output file if exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Set up seed
    set_seed(SEED)

    # Load data
    extension = data_file.split(".")[-1]
    if extension == "txt":
        extension = "text"

    dataset = load_dataset(extension, data_files=data_path)["train"]

    # Sample data for debugging
    if DEBUG:
        dataset = dataset.select(range(300))

    # Load tokenizers
    tokenizers_names = os.listdir(tokenizers_dir)

    results = {}
    # iterate over tokenizers
    for tokenizer_name in tqdm(tokenizers_names, desc="Calculating CTC"):
        tokenizer_path = tokenizers_dir + tokenizer_name
        # Calculate CTC
        ctc, comp_rate = calc_metrics(tokenizer_path, dataset)
        results[tokenizer_name] = {"ctc": ctc, "compression_rate": comp_rate}

        # If BERT, calculate CTC for Flota version
        if "bert" in tokenizer_name:
            ctc_flota, comp_rate_flota = calc_metrics(
                tokenizer_path, dataset, flota=True
            )
            results[tokenizer_name]["ctc_flota_k_10"] = ctc_flota
            results[tokenizer_name]["compression_rate_flota_k_10"] = comp_rate_flota
            # Calculate Ratio
            results[tokenizer_name]["ratio_k_10"] = ctc_flota / ctc

            ctc_flota, comp_rate_flota = calc_metrics(
                tokenizer_path, dataset, flota=True, k=2
            )
            results[tokenizer_name]["ctc_flota_k_2"] = ctc_flota
            results[tokenizer_name]["compression_rate_flota_k_2"] = comp_rate_flota
            # Calculate Ratio
            results[tokenizer_name]["ratio_k_2"] = ctc_flota / ctc


    results = pd.DataFrame(results).T

    # Save results
    results.to_csv(output_file)


if __name__ == "__main__":
    main()
