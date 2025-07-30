"""
This module contains the function for running domain adaptation.
"""

import os
import logging
from argparse import Namespace
from itertools import product
from typing import Callable
from datasets import load_dataset

from domain_adaptation.simple import simple_domain_adaptation
from domain_adaptation.ada_lm import ada_lm_domain_adaptation
from domain_adaptation.idf_da import idf_adaptation


def run_domain_adaptation(
    args: Namespace, get_model_name_or_path: Callable[[str], str]
):
    """
    Run domain adaptation.
    :param args: The arguments for domain adaptation.
    :param get_model_name_or_path: A function that returns the model name or path for a given model base name.
    return: A list of dictionaries with information about the models used for domain adaptation.
    """

    # Set up logging
    logger = logging.getLogger(__name__)

    # Set train path
    data_dir = "further_pre_training/data"
    if not args.corpus_path:
        if args.corpus_name == "doctors":
            args.corpus_path = os.path.join(data_dir, "doctors.txt")
        elif args.corpus_name == "demo":
            args.corpus_path = os.path.join(data_dir, "demo.txt")
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    # Get data
    extension = args.corpus_path.split(".")[-1]
    if extension == "txt":
        extension = "text"
    dataset = load_dataset(extension, data_files=args.corpus_path)["train"]
    logger.info(f"Loaded corpus from {args.corpus_path}")


    # Do debugging
    if args.debug:
        dataset = dataset.select(range(100))
        logger.info("Using a subset of the data for debugging")
    elif args.train_data_size:
        args.train_data_size = min(int(args.train_data_size), len(dataset))
        # Shuffle and sample train_data_size examples
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(args.train_data_size))
        logger.info(f"Selected {args.train_data_size} examples for training with seed {args.seed}")
    else:
        logger.info("Using the full dataset for training")
    
    # Do basic preprocessing - strip, lower
    dataset = dataset.map(lambda example: {"text": example["text"].strip().lower()}, desc="Basic preprocessing")
    logger.info("Preprocessed corpus")

    # Define the DAs to perform
    models_info = []
    combinations = product(args.tokenizer_base_names, args.schemas)
    for tokenizer_base_name, schema in combinations:
        model_info = {
            "short_name": tokenizer_base_name,
            "base_ckpt": get_model_name_or_path(tokenizer_base_name),
        }
        TOKENIZER_NAME = f"{args.corpus_name}_{model_info['short_name']}_{schema}_seed{args.seed}"

        logger.info(
            f"Performing domain adaptation with {tokenizer_base_name} and schema {schema}"
        )

        if schema == "simple":
            # Reset vocab size
            args.vocab_size = args.simple_vocab_size
            # Adapt vocab size for debugging
            if args.debug:
                args.vocab_size = 1000

            # Adapt the tokenizer name
            TOKENIZER_NAME += f"_{int(args.vocab_size / 1000)}k"
            tokenizer_adaptation_fn = simple_domain_adaptation
            kwargs = {
                "vocab_size": args.vocab_size,
                "scratch": False
            }
        elif schema == "ada_lm":
            args.interval = args.ada_lm_interval
            tokenizer_adaptation_fn = ada_lm_domain_adaptation
            kwargs = {
                "interval": args.ada_lm_interval,
                "th": args.ada_lm_th,
            }
        elif schema == "scratch":
            # Vocab size as the base model
            kwargs = {
                "vocab_size": None,
                "scratch": True
            }
            tokenizer_adaptation_fn = simple_domain_adaptation
        elif schema == "avocado":
            # TODO: implement avocado schema
            raise NotImplementedError("avocado schema is not implemented yet")
        else:
            raise ValueError(f"Schema {args.schema} not found")

        # Set the output directory and check if it exists
        model_info["tokenizer"] = os.path.join(args.output_dir, TOKENIZER_NAME)
        if os.path.exists(model_info["tokenizer"]) and not args.overwrite:
            logger.info(f'Tokenizer {model_info["tokenizer"]} already exist!')
        else:
            logger.info(
                f'Tokenizer {model_info["tokenizer"]} does not exist yet, starting training'
            )
            tokenizer_adaptation_fn(dataset, model_info, args, **kwargs)
            logger.info(f'Finished training of tokenizer: {model_info["tokenizer"]}')

        # Do idf adaptation if needed
        if args.do_idf:
            logger.info(f"Performing IDF adaptation for {model_info['tokenizer']}")
            idf_tokenizer_name = model_info["tokenizer"] + "_idf"
            idf_tokenizer_name += "_rm_prefix" if args.idf_rm_prefix else ""
            if os.path.exists(idf_tokenizer_name) and not args.overwrite:
                logger.info(
                    f'Tokenizer {idf_tokenizer_name} already exist!'
                )
                model_info["tokenizer"] = idf_tokenizer_name
            else:
                kwargs = {
                    "vocab_size": args.idf_vocab_size,
                    "max_score": args.idf_max_score,
                    "count_th": args.idf_count_th,
                    "rm_prefix": args.idf_rm_prefix,
                }
                model_info = idf_adaptation(dataset, model_info, args, **kwargs)
                logger.info(
                    f'Finished training of tokenizer: {model_info["tokenizer"]}'
                )

        models_info.append(model_info)
    return models_info
