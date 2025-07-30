# -*- coding: utf-8 -*-

"""
A module with utility functions for running Domain Adaptation.
"""

import re
from datasets import Dataset
from transformers import AutoTokenizer

# Define global variables
CODE_REMOVER = re.compile(
    "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＠。、？！｀＋￥％0-9]+"
)


def _get_training_corpus(dataset: Dataset):
    """
    Get a training corpus from the dataset as a generator.
    :param dataset: The dataset to get the training corpus from.
    :return: A generator with the training corpus.
    """
    return (dataset[i : i + 1000]["text"] for i in range(0, len(dataset), 1000))


def train_tokenizer(
    dataset: Dataset, vocab_size: int, base_tokenizer: AutoTokenizer = None
):
    """
    Train a tokenizer.
    :param corpus: The corpus to use for training the tokenizer.
    :param vocab_size: The vocabulary size to use for training the tokenizer.
    :param base_tokenizer: The base tokenizer to use for training the tokenizer.
    :return: The trained tokenizer.
    """
    # Create a training corpus
    corpus = _get_training_corpus(dataset)

    # Train the tokenizer
    tokenizer = base_tokenizer.train_new_from_iterator(
        corpus, vocab_size, show_progress=True
    )
    return tokenizer


def adapt_tokenizer(
    base_tokenizer: AutoTokenizer, new_vocab: list[str]
) -> tuple[AutoTokenizer, dict[str, int]]:
    """
    Adapt a tokenizer by adding a new vocabulary to it.
    :param base_tokenizer: The base tokenizer to adapt.
    :param new_vocab: The new vocabulary to add to the tokenizer.
    :return: The adapted tokenizer and a dictionary with the added tokens and their indices.
    """
    base_added_tokens = base_tokenizer.get_added_vocab()
    base_vocab = base_tokenizer.get_vocab().keys()

    adapted_tokenizer = base_tokenizer
    # Get the tokens to add
    tokens_to_add = list(set(new_vocab) - set(base_vocab))
    # Add the tokens to the tokenizer
    adapted_tokenizer.add_tokens(tokens_to_add)
    # Get a dictionary of the added tokens and their indices
    added_tokens = adapted_tokenizer.get_added_vocab()
    for token in base_added_tokens:
        added_tokens.pop(token)

    return adapted_tokenizer, added_tokens


def clean_vocab(vocab: list[str]) -> list[str]:
    """
    Clean a vocabulary by removing noise tokens.
    :param vocab: The vocabulary to clean.
    :return: The cleaned vocabulary.
    """
    vocab = list(filter(lambda token: not CODE_REMOVER.match(token), vocab))

    # Clean non printable characters
    vocab = list(filter(lambda token: token.isprintable(), vocab))
    return vocab
