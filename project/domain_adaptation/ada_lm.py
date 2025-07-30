"""
A module to run domain adaptation using the AdaLM schema.
"""

import os
import json
import logging
from argparse import Namespace

import numpy as np
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re


from datasets import Dataset
from transformers import AutoTokenizer

from domain_adaptation.utils import train_tokenizer, adapt_tokenizer


# Define global variables
CODE_REMOVER = re.compile(
    "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＠。、？！｀＋￥％0-9]+"
)


def compute_counter(tokenized_dataset: Dataset) -> Counter:
    """
    Compute the counter of a tokenized dataset.
    :param tokenized_dataset: the tokenized dataset.
    :return: the counter of the tokenized dataset.
    """

    all_input_ids = list(chain(*tokenized_dataset["input_ids"]))
    counter = Counter(all_input_ids)
    return counter


def tokenize_corpus(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Tokenize a corpus given a tokenizer.
    :param dataset: the corpus.
    :param tokenizer: a tokenizer instance to evaluate.
    :return: the tokenized corpus.
    """

    # Define tokenization function
    def tokenize_function(examples, tokenizer=tokenizer):
        return tokenizer(examples["text"], truncation=True)

    # Tokenize data
    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing")
    return tokenized_dataset


def compute_language_model(dataset: Dataset, tokenizer: AutoTokenizer) -> float:
    """
    Compute the mean P(D) of a corpus given a tokenizer.
    :param dataset: the corpus.
    :param tokenizer: a tokenizer instance to evaluate.
    :return: P(D).
    """
    tokenized_dataset = tokenize_corpus(dataset, tokenizer)

    # Count tokens and tokens appearance
    counter_dict = dict(compute_counter(tokenized_dataset))
    all_tokens = sum(counter_dict.values())

    # Calculate p(x_i)
    for token in counter_dict.keys():
        counter_dict[token] /= all_tokens

    # Calculate P(X)
    tokenized_dataset = tokenized_dataset.map(
        lambda example: {
            "prob": sum(np.log(counter_dict[token]) for token in example["input_ids"])
        },
        desc="Calculating P(D)",
    )

    # Calculate P(D)
    p_d = np.mean(tokenized_dataset["prob"])
    return p_d


def get_added_vocab(
    dataset: Dataset, tokenizer: AutoTokenizer, present_vocab: set, increment: int
) -> list[str]:
    """
    Get the added vocabulary of a tokenizer.
    :param dataset: the dataset to use for domain adaptation.
    :param tokenizer: the tokenizer to use for domain adaptation.
    :param present_vocab: the present vocabulary of the tokenizer.
    :param increment: the increment of the vocabulary size.
    :return: the added vocabulary of the tokenizer.
    """
    # Tokenize and count the dataset
    tokenized_dataset = tokenize_corpus(dataset, tokenizer)
    counter = compute_counter(tokenized_dataset)

    # Get the added vocabulary and decode it to strings
    added_vocab = []
    for k, _ in counter.most_common():
        token = tokenizer.decode([k])
        if token not in present_vocab:
            added_vocab.append(token)

    # Post-process the added vocabulary
    new_added_vocab = []
    for av in tqdm(added_vocab):
        if CODE_REMOVER.match(av):
            continue
        new_added_vocab.append(av)

        # Break if we have enough tokens
        if len(new_added_vocab) == increment:
            break
    return new_added_vocab


def ada_lm_domain_adaptation(
    dataset: Dataset, model_info: dict, args: Namespace, **kwargs
):
    """
    Perform domain adaptation using the simple schema.
    Save the adapted tokenizer to the output directory.
    :param dataset: The dataset to use for domain adaptation.
    :param model_info: A dictionary with information about the model.
    :param args: The arguments for domain adaptation.
    :param interval: The interval of the vocabulary size.
    :param th: The final threshold of the P(D)'s increase
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Get parameters
    interval = kwargs["interval"]
    th = kwargs["th"]

    # ATM supports only wordpiece
    alg = "wordpiece"

    adapted_tokenizer_path = model_info["tokenizer"]

    # Load the present tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_info["base_ckpt"])
    present_tokenizer = last_tokenizer = AutoTokenizer.from_pretrained(
        model_info["base_ckpt"]
    )
    # Get base vocab
    orig_vocab = last_vocab = set(base_tokenizer.get_vocab().keys())
    # Get original size
    orig_size = len(orig_vocab)
    # Calculate P of the base tokenizer
    last_p = compute_language_model(dataset, present_tokenizer)

    target_size = orig_size

    # Iteratively build new tokenizer
    delta = th + 1

    all_added_tokens = {}
    logger.info(f"Original size: {orig_size} with P(D) of {round(last_p, 4)}")
    while delta > th:
        # Update vocab target size
        target_size += interval
        logger.info(f"Target size: {target_size}")
        DOMAIN_TOKENIZER_NAME = f"{alg}_seed{args.seed}_{target_size}"
        domain_tokenizer_path = os.path.join(args.output_dir, DOMAIN_TOKENIZER_NAME)

        # Train a domain specific tokenizer
        if os.path.exists(domain_tokenizer_path):
            present_tokenizer = AutoTokenizer.from_pretrained(domain_tokenizer_path)
        else:
            present_tokenizer = train_tokenizer(
                dataset, target_size, base_tokenizer=base_tokenizer
            )
            present_tokenizer.save_pretrained(domain_tokenizer_path)

        # Get the added tokens
        added_tokens = get_added_vocab(dataset, present_tokenizer, last_vocab, interval)
        # Merge new tokens with the last tokenizer
        present_tokenizer, added_tokens = adapt_tokenizer(last_tokenizer, added_tokens)
        all_added_tokens.update(added_tokens)

        # Calculate the current's tokenizer probability
        cur_p = compute_language_model(dataset, present_tokenizer)
        logger.info(
            f"Current size: {len(present_tokenizer)} with P(D) of {round(cur_p, 4)}"
        )
        logger.info(f"Got {len(added_tokens)} new tokens")

        delta = (last_p - cur_p) / last_p
        logger.info(f"Current delta: {round(delta, 4)}")

        last_p = cur_p
        last_tokenizer = present_tokenizer
        last_vocab = set(last_tokenizer.get_vocab().keys())

    # Iteration converged, merge the new tokenizer with the base tokenizer
    adapted_tokenizer = present_tokenizer
    logger.info(
        f"Iteration done. Adapted tokenizer with {len(all_added_tokens)} new tokens, new vocab size: {len(adapted_tokenizer)}"
    )

    # Save the merged tokenizer and the added tokens
    adapted_tokenizer.save_pretrained(adapted_tokenizer_path)
    added_tokens_path = os.path.join(adapted_tokenizer_path, "added_tokens_dict.json")
    with open(added_tokens_path, "w", encoding="utf8") as f:
        json.dump(all_added_tokens, f, indent=2, ensure_ascii=False)
