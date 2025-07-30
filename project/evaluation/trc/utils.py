"""
A module with utility functions and the main function for running a TRC experiment.
"""

import logging
from argparse import Namespace
from collections import Counter


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


from datasets import DatasetDict
from transformers import AutoTokenizer

from evaluation.trc.trc_model.trc_model import TRCBert, TRCRoberta
from evaluation.trc.trc_model.trc_config import TRCBertConfig, TRCRobertaConfig

from further_pre_training.utils import FlotaBERTTokenizer

# Define global variables
TRC_GLOBAL_VARIABLES = {
    "LABELS": ["BEFORE", "AFTER", "EQUAL", "VAGUE"],
    "LABELS_NO_VAGUE": ["BEFORE", "AFTER", "EQUAL"],
    "LABELS_IDS": [0, 1, 2, 3],
    "columns": [
        "MODEL",
        "ESS_mean",
        "ESS_std",
        "EMP_mean",
        "EMP_std",
        "SEQ_CLS_mean",
        "SEQ_CLS_std",
    ],
}

E1_start, E1_end, E2_start, E2_end = "[א1]", "[/א1]", "[א2]", "[/א2]"


def calculate_class_weights(dataset: DatasetDict) -> list[float]:
    labels = dataset["train"]["label"]
    labels_count = Counter(labels)
    class_weights = [0] * len(labels_count)
    for label, count in labels_count.items():
        cls_w = 1 - (count / len(labels))
        class_weights[label] = cls_w
    return class_weights


def trc_compute_metrics(preds):
    predictions, labels = preds
    predictions = np.argmax(predictions, axis=1)

    results = classification_report(
        labels,
        predictions,
        output_dict=True,
        target_names=TRC_GLOBAL_VARIABLES["LABELS"],
    )
    final_results = results["weighted avg"]
    final_results.pop("support")
    final_results["BEFORE-f1"] = results["BEFORE"]["f1-score"]
    final_results["AFTER-f1"] = results["AFTER"]["f1-score"]
    final_results["EQUAL-f1"] = results["EQUAL"]["f1-score"]
    final_results["VAGUE-f1"] = results["VAGUE"]["f1-score"]

    return final_results


def trc_evaluate(
    predictions,
    labels,
    output_dir: str = None,
    logger: logging.Logger = None,
    **kwargs,
):
    id2label = kwargs["id2label"]
    report = classification_report(
        labels, predictions, target_names=TRC_GLOBAL_VARIABLES["LABELS"]
    )

    cm = confusion_matrix(labels, predictions)
    unique_labels = np.unique(np.concatenate((labels, predictions)))
    class_names = [id2label[i] for i in unique_labels]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    logger.info(f"Confusion Matrix with vague :\n{cm_df.to_string()}\n")

    # Create a report without mistakes on the 'VAGUE' class
    for i in range(len(labels)):
        if labels[i] == 3 and predictions[i] != 3:
            labels[i] = predictions[i]

    report_no_vague_str = classification_report(
        labels,
        predictions,
        target_names=TRC_GLOBAL_VARIABLES["LABELS"],
        labels=TRC_GLOBAL_VARIABLES["LABELS_IDS"],
    )
    report_no_vague = classification_report(
        labels,
        predictions,
        target_names=TRC_GLOBAL_VARIABLES["LABELS"],
        output_dict=True,
        labels=TRC_GLOBAL_VARIABLES["LABELS_IDS"],
    )
    logger.info(f"Evaluation report:\n{report}\n")
    logger.info(f"Evaluation report without VAGUE class:\n{report_no_vague_str}\n")

    
    

    # Extract the weighted average f1 score of the no-vague report
    weighted_avg_f1 = report_no_vague["weighted avg"]["f1-score"]

    return -1, weighted_avg_f1


def trc_prepare_for_training(
    ckpt: str,
    raw_datasets: DatasetDict,
    label2id: dict[str, int],
    id2label: dict[int, str],
    logger: logging.Logger,
    args: Namespace,
    class_weights: list[float],
    arc: str = "ESS",
):
    # Load and add special tokens to tokenizer
    use_fast = False if args.flota else True
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=use_fast)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[א1]", "[/א1]", "[א2]", "[/א2]"]}
    )
    E1_start = tokenizer.convert_tokens_to_ids("[א1]")
    E1_end = tokenizer.convert_tokens_to_ids("[/א1]")
    E2_start = tokenizer.convert_tokens_to_ids("[א2]")
    E2_end = tokenizer.convert_tokens_to_ids("[/א2]")

    # Apply Flota if needed
    if args.flota:
        tokenizer.wordpiece_tokenizer = FlotaBERTTokenizer(tokenizer)

    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenizer_class = str(type(tokenizer)).strip("><'").split(".")[-1]

    logger.info("Tokenized datasets successfully")

    # Define the model's base architecture
    if args.model_arc == "bert":
        config_class = TRCBertConfig
        model_class = TRCBert
    else:
        config_class = TRCRobertaConfig
        model_class = TRCRoberta

    # Define model and trainer
    config = config_class(
        EMS1=E1_start,
        EMS2=E2_start,
        EME1=E1_end,
        EME2=E2_end,
        class_weights=class_weights,
        architecture=arc,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        name_or_path=ckpt,
        tokenizer_class=tokenizer_class,
        vocab_size=len(tokenizer),
    )

    model = model_class(config=config)

    # If probing, freeze the model besides the classifier layer
    if args.probing:
        # Train trc related layers and the classifier layer
        layers_to_train = [
            "classifier",
            "relation_representation",
            "post_transformer",
            "post_transformer_1",
            "post_transformer_2",
            "classification_layer"
        ]
        logger.info("Freezing base model layers")
        for name, param in model.named_parameters():
            if not any([layer in name for layer in layers_to_train]):
                param.requires_grad = False

    return model, tokenized_datasets, tokenizer, None