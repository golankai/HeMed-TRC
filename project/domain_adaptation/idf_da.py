"""
A module to run domain adaptation based on IDF scores
Will calculate IDF for the corpus and merge high scores words to  the base tokenizer.
This adaptation comes on top of another adaptation.
"""

import os
import json
import logging
from argparse import Namespace
import re
import numpy as np
from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer

from domain_adaptation.utils import adapt_tokenizer, clean_vocab

# Define global variables
# fmt: off
ALL_MODERN_HEBREW_PREFIXES = set(["ב", "בכ", "בש", "בשה", "בשל", "ה", "ו", "כ", "כב", "כש", "כשב", "כשה", "כשכ", "כשל", "כשמ", "כשמה", "ל", "לכ", "לכש", "לכשה", "למ", "למב", "למש", "לש", "לשל", "מ", "מב", "מבש", "מה", "מכ", "מכש", "מל", "מלש", "מש", "משה", "משל", "ש", "שב", "שה", "שכ", "שכש", "שכשה", "של", "שלכש", "שלכשה", "שלש", "שמ", "שמב", "שמה", "שמש", "וב", "ובכ", "ובש", "ובשה", "ובשל", "וה", "וכ", "וכב", "וכש", "וכשב", "וכשה", "וכשכ", "וכשל", "וכשמ", "וכשמה", "ול", "ולכ", "ולכש", "ולכשה", "ולמ", "ולמב", "ולמש", "ולש", "ולשל", "ומ", "ומב", "ומבש", "ומה", "ומכ", "ומכש", "ומל", "ומלש", "ומש", "ומשה", "ומשל", "וש", "ושב", "ושה", "ושכ", "ושכש", "ושכשה", "ושל", "ושלכש", "ושלכשה", "ושלש", "ושמ", "ושמב", "ושמה", "ושמש"])
# fmt: on

PREFIX_REGEX = (
    f'(^|\\s)({"|".join(ALL_MODERN_HEBREW_PREFIXES)})(?![א-ת](?:[א-ת]{{0,3}}\\s|$))'
)

ALL_MODERN_HEBREW_SUFFIXES = set(["ות", "ים"])
SUFFIX_REGEX = f'({"|".join(ALL_MODERN_HEBREW_SUFFIXES)})(?![א-ת])'

AFFIX_REGEX = f"{PREFIX_REGEX}|{SUFFIX_REGEX}"


def calculate_idf(dataset: Dataset, rm_prefix: bool = False, count_th : int = 0, logger = None) -> dict[str, float]:
    """
    Calculate the IDF scores for the dataset based on words.
    
    :param dataset: The dataset to calculate IDF scores for.
    :param rm_prefix: If True, remove prefixes from words before calculating IDF.
    :param count_th: The minimum number of occurrences for a word to be considered.
    :param logger: The logger to use for logging.
    :return: A dictionary with the IDF scores for the dataset.
    """
    total_docs = len(dataset)
    
    # Use Counter to efficiently count word occurrences across documents
    idf_counts = Counter()
    
    def tokenize_and_count(example):
        text = example["text"]
        
        if rm_prefix:
            # Remove prefixes to get "roots" of the words
            text_no_prefixes = re.sub(PREFIX_REGEX, " ", text)
            text += " " + text_no_prefixes
        
        # Tokenize the text into words (not characters)
        words = set(text.split())
        idf_counts.update(words)
        return example

    # Apply tokenization and count on the dataset
    _ = dataset.map(tokenize_and_count, batched=False, desc="Tokenizing and counting")

    # Leave only tokens that occur enough times
    logger.info(
        f"Vocab size before filtering by occurrences threshold: {len(idf_counts)}"
    )
    idf_counts = {k: v for k, v in idf_counts.items() if v >= count_th}
    logger.info(
        f"Vocab size after filtering by occurrences threshold: {len(idf_counts)}"
    )

    # Convert counts to a NumPy array for vectorized operations
    words = list(idf_counts.keys())
    counts = np.array(list(idf_counts.values()), dtype=np.float64)
    
    # Compute IDF scores using NumPy vectorized operations
    idf_scores = np.log(total_docs / counts)
    
    # Create a dictionary with words and their IDF scores
    idf_dict = dict(zip(words, idf_scores))
    
    # Sort the IDF scores by score in descending order
    sorted_idf_scores = dict(sorted(idf_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_idf_scores


def idf_adaptation(dataset: Dataset, model_info: dict, args: Namespace, **kwargs):
    """
    Perform domain adaptation using the IDF schema.
    Save the adapted tokenizer to the output directory.
    :param dataset: The dataset to use for domain adaptation.
    :param model_info: A dictionary with information about the model.
    :param args: The arguments for domain adaptation.
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Load the previous tokenizer and the added tokens
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    with open(
        os.path.join(model_info["tokenizer"], "added_tokens_dict.json"),
        "r",
        encoding="utf-8",
    ) as f:
        prev_added_tokens = json.load(f)

    # Define the paths for the domain vocab and the adapted tokenizer
    model_info["tokenizer"] = model_info["tokenizer"] + "_idf"
    model_info["tokenizer"] += "_rm_prefix" if kwargs["rm_prefix"] else ""
    DOMAIN_VOCAB_NAME = f"{args.corpus_name}_idf_seed{args.seed}"
    DOMAIN_VOCAB_NAME += "_rm_prefix" if kwargs["rm_prefix"] else ""
    DOMAIN_VOCAB_NAME += ".json"
    domain_vocab_path = os.path.join(args.output_dir, DOMAIN_VOCAB_NAME)
    adapted_tokenizer_path = model_info["tokenizer"]

    # Create IDF scores for the dataset
    if os.path.exists(domain_vocab_path) and not args.overwrite:
        logger.info(f"Domain vocab {domain_vocab_path} already exists")
        with open(domain_vocab_path, "r", encoding="utf-8") as f:
            domain_vocab = json.load(f)
    else:
        logger.info(f"Training domain IDF vocab {domain_vocab_path}")
        # Calculate the IDF scores for the dataset
        domain_vocab = calculate_idf(dataset, kwargs["rm_prefix"], kwargs["count_th"], logger)
        # Clean the vocabulary
        clean_tokens = clean_vocab(domain_vocab.keys())
        domain_vocab = {k: v for k, v in domain_vocab.items() if k in clean_tokens}

        # Save the domain vocab to a json file
        with open(domain_vocab_path, "w", encoding="utf8") as f:
            json.dump(domain_vocab, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved domain vocab to {domain_vocab_path}")

    # Define the new vocabulary to add to the base tokenizer
    if "vocab_size" in kwargs and kwargs["vocab_size"]:
        vocab_size = int(kwargs["vocab_size"])
        new_vocab = list(domain_vocab.keys())[:vocab_size]
    elif "max_score" in kwargs and kwargs["max_score"]:
        max_score = kwargs["max_score"]
        domain_vocab = {k: v for k, v in domain_vocab.items() if v < max_score}
        new_vocab = list(domain_vocab.keys())
    else:
        # Already extracted the vocab based on the count threshold
        new_vocab = list(domain_vocab.keys())

    # Merge the domain specific tokenizer with the base tokenizer
    adapted_tokenizer, idf_added_tokens = adapt_tokenizer(tokenizer, new_vocab)
    logger.info(
        f"Adapted tokenizer with {len(idf_added_tokens)} new tokens, new vocab size: {len(adapted_tokenizer)}"
    )

    # Save the merged tokenizer and the added tokens
    adapted_tokenizer.save_pretrained(adapted_tokenizer_path)

    # Save the IDF added tokens to a json file
    idf_added_tokens_path = os.path.join(
        adapted_tokenizer_path, "idf_added_tokens_dict.json"
    )
    with open(idf_added_tokens_path, "w", encoding="utf8") as f:
        json.dump(idf_added_tokens, f, indent=2, ensure_ascii=False)

    # Save all the added tokens to a json file
    added_tokens = {**prev_added_tokens, **idf_added_tokens}
    added_tokens_path = os.path.join(adapted_tokenizer_path, "added_tokens_dict.json")
    with open(added_tokens_path, "w", encoding="utf8") as f:
        json.dump(added_tokens, f, indent=2, ensure_ascii=False)

    return model_info
