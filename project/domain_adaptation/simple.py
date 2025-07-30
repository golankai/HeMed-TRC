"""
A module to run simple domain adaptation.
Will train a domain specific tokenizer and merge it with the base tokenizer.
"""

import os
import json
import logging
from argparse import Namespace
from datasets import Dataset
from transformers import AutoTokenizer

from domain_adaptation.utils import train_tokenizer, adapt_tokenizer


def simple_domain_adaptation(
    dataset: Dataset, model_info: dict, args: Namespace, **kwargs
):
    """
    Perform domain adaptation using the simple schema.
    Save the adapted tokenizer to the output directory.
    :param dataset: The dataset to use for domain adaptation.
    :param model_info: A dictionary with information about the model.
    :param args: The arguments for domain adaptation.
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Load the base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_info["base_ckpt"])

    # Get the vocab size
    vocab_size = kwargs["vocab_size"]

    # If vocab size is not provided, use the base tokenizer's vocab size
    if vocab_size is None:
        vocab_size = len(base_tokenizer)
    DOMAIN_TOKENIZER_NAME = f"{model_info['short_name']}_seed{args.seed}_{vocab_size}"
    domain_tokenizer_path = os.path.join(args.output_dir, DOMAIN_TOKENIZER_NAME)
    adapted_tokenizer_path = model_info["tokenizer"]

    # Train a domain specific tokenizer
    if os.path.exists(domain_tokenizer_path) and not args.overwrite:
        logger.info(f"Domain tokenizer {domain_tokenizer_path} already exists")
        domain_tokenizer = AutoTokenizer.from_pretrained(domain_tokenizer_path)
    else:
        logger.info(f"Training domain tokenizer {domain_tokenizer_path}")
        domain_tokenizer = train_tokenizer(
            dataset, vocab_size, base_tokenizer=base_tokenizer
        )
        logger.info(f"Type of domain_tokenizer: {type(domain_tokenizer)}")
        domain_tokenizer.save_pretrained(domain_tokenizer_path)
        logger.info(f"Saved domain tokenizer to {domain_tokenizer_path}")

    if kwargs["scratch"]:
        adapted_tokenizer = domain_tokenizer
        adapted_tokenizer.save_pretrained(adapted_tokenizer_path)

    else:
        # Define the new vocabulary to add to the base tokenizer
        new_vocab = list(domain_tokenizer.get_vocab().keys())

        # Merge the domain specific tokenizer with the base tokenizer
        adapted_tokenizer, added_tokens = adapt_tokenizer(base_tokenizer, new_vocab)
        logger.info(
            f"Adapted tokenizer with {len(added_tokens)} new tokens, new vocab size: {len(adapted_tokenizer)}"
        )

        # Save the merged tokenizer and the added tokens
        adapted_tokenizer.save_pretrained(adapted_tokenizer_path)
        added_tokens_path = os.path.join(adapted_tokenizer_path, "added_tokens_dict.json")
        with open(added_tokens_path, "w", encoding="utf8") as f:
            json.dump(added_tokens, f, indent=2, ensure_ascii=False)
