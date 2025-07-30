"""
A module with utility functions and the main function for running an evaluation.
"""

import os
from shutil import rmtree
from copy import deepcopy
import logging
from argparse import Namespace
import re
from collections import Counter

import numpy as np
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)
from torch.cuda import current_device, is_available
from sklearn.metrics import confusion_matrix
import evaluate

from evaluation.trc.utils import (
    TRC_GLOBAL_VARIABLES,
    trc_compute_metrics,
    calculate_class_weights,
    trc_prepare_for_training,
    trc_evaluate,
)

seqeval_metric = evaluate.load("seqeval")

GLOBAL_VARIABLES = {
    "trc": TRC_GLOBAL_VARIABLES,
}


class ContiguousTrainer(Trainer):
    def _save(self, output_dir):
        # Ensure all tensors are contiguous before saving
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        super()._save(output_dir)


def _run_eval_wrapper(
    model_info,
    args,
    logger,
    seeds,
    raw_datasets,
    label2id,
    id2label,
    arc=None,
    class_weights=None,
    compute_metrics=None,
    prepare_for_training=None,
    evaluate=None,
    **kwargs,
):
    # Set model name
    if not model_info["from_hf"]:
        # Done further pretraining (and domain adaptation), so use the pretrained model name
        model_name_results = re.split(r"[\\/]", model_info["trained_ckpt"])[-1]
    else:
        # Running on a model that has not been further pre-trained, so use the model name
        model_name_results = model_info["short_name"]

    # Add proportion to model name if less than 1
    if args.proportion < 1:
        model_name_results += f"_eval_prop_{int(args.proportion * 100)}"
    
    # Add max_train_samples to model name if needed
    if args.max_train_samples:
        model_name_results += f"_eval_train_samples_{args.max_train_samples}"

    # Add flota to model name if needed
    if args.flota or args.auto_use_flota_eval and "flota" in model_name_results:
        model_name_results += "_flota"
        args.flota = True
        logger.info("Using Flota for evaluation")
    else:
        args.flota = False
        logger.info("Not using Flota for evaluation")

    MODEL_NAME = model_name_results
    if arc:
        # Add architecture to model name
        MODEL_NAME += f"_{arc}"

    logger.info(f"Training model: {MODEL_NAME}")

    acc_scores = []
    f1_scores = []

    for seed in seeds:
        cur_dataset = deepcopy(raw_datasets)
        # Set seed
        args.cur_seed = seed
        set_seed(seed)
        cur_name = MODEL_NAME + f"_{seed}"
        cur_output_dir = os.path.join(args.output_dir, cur_name)

        # Truncate data if needed
        if args.new_len:
            # If need to truncate data, do so
            cur_dataset["train"] = cur_dataset["train"].shuffle(seed=seed)
            logger.info(f"Truncated data from {len(cur_dataset['train'])} to {args.new_len} with seed {seed}")
            cur_dataset["train"] = cur_dataset["train"].select(range(args.new_len))

        if "label" in cur_dataset["train"].features:
            logger.info(f"Train set label distribution (seed {seed}): {Counter(cur_dataset['train']['label'])}")

        model, tokenized_datasets, tokenizer, data_collator = prepare_for_training(
            model_info["trained_ckpt"],
            cur_dataset,
            label2id,
            id2label,
            logger,
            args,
            arc=arc,
            class_weights=class_weights,
        )


        training_args = TrainingArguments(
            output_dir=cur_output_dir,
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=0.01,
            warmup_steps=100 if not args.cur_task == "trc" else 0,
            num_train_epochs=args.epochs,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=args.logging_steps,
            save_steps=args.logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1" if not args.cur_task == "trc" else "loss",
            report_to=["tensorboard"],
            seed=args.cur_seed,
            logging_dir=os.path.join(cur_output_dir, "tb_logs"),
            logging_steps=args.logging_steps,
        )

        trainer = ContiguousTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train
        trainer.train()
        logger.info("Finished training")

        # Save model
        trainer.save_model(cur_output_dir)
        logger.info(f"Model saved at {cur_output_dir}")

        # Evaluate on dev
        results = trainer.evaluate()
        logger.info(f"Results on dev set: {results}")

        # Evaluate on test
        predictions, labels, metrics = trainer.predict(tokenized_datasets["test"])
        logger.info(metrics)
       
        predictions = np.argmax(predictions, axis=1)
        predictions = predictions

        acc, f1 = evaluate(
            predictions,
            labels,
            metrics=metrics,
            output_dir=cur_output_dir,
            logger=logger,
            id2label=id2label,
        )

        logger.info(f"Test predicted Label Distribution: {Counter(predictions)}")

        acc_scores.append(acc)
        f1_scores.append(f1)

        # Log a confusion matrix
        cm = confusion_matrix(labels, predictions)
        # Filter out -100 (or other invalid labels)
        unique_labels = [label for label in np.unique(np.concatenate((labels, predictions))) if label != -100]
        class_names = [id2label[i] for i in unique_labels]
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        logger.info(f"Confusion Matrix:\n{cm_df.to_string()}\n")
        logger.info(f"Finished training model {MODEL_NAME} with seed {seed}")

        # If need to delete the model, do so
        if args.delete_after_training:
            rmtree(cur_output_dir)

    return acc_scores, f1_scores, model_name_results


def run_eval(args: Namespace, models_info: list[dict[str, str]]):
    """
    Run an evaluation round
    :param args: the arguments for the experiment
    :param models_info: a list of dictionaries with information about the models to use for TRC
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    # Set up seeds
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    
    orig_args = deepcopy(args)
    for task in args.eval_tasks:
        args = deepcopy(orig_args)
        args.cur_task = task
        # Set up global variables
        variables = GLOBAL_VARIABLES[task]
        # Set up the data, results and output directory
        args.output_dir = os.path.join(f"evaluation/{task}", "models")
        args.data_path = os.path.join("evaluation", task, "data")
        args.results_path = os.path.join("evaluation", task, "results")
        if task == "trc":
            args.data_path = os.path.join(args.data_path, args.trc_dataset_name)
            args.results_path = os.path.join(args.results_path, args.trc_dataset_name)
            args.learning_rate = args.trc_learning_rate
        else:
            raise ValueError(f"Task {task} not supported")

        # If saving results, make sure directory exists
        if args.update_existing_results:
            os.makedirs(args.results_path, exist_ok=True)
            if args.probing:
                args.results_path = os.path.join(args.results_path, "results_probing.csv")
            else:
                args.results_path = os.path.join(args.results_path, "results.csv")

        # Set up some task-specific functions and details
        if task == "trc":
            compute_metrics = trc_compute_metrics
            prepare_for_training = trc_prepare_for_training
            evaluate = trc_evaluate
        else:
            raise ValueError(f"Task {task} not supported")
      

        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Start {task} training")

        if args.update_existing_results and os.path.exists(args.results_path):
            existing_results_df = pd.read_csv(args.results_path, index_col="MODEL")
        else:
            existing_results_df = None

        # Create a new df for this run
        results_df = pd.DataFrame(columns=variables["columns"])
        results_df.set_index("MODEL", inplace=True)

        # Get data
        raw_datasets = load_dataset(args.data_path)
        logger.info(f"Loaded data from  {args.data_path} successfully")

        # Re-split if needed
        if args.split_seed:
            raw_datasets = concatenate_datasets(
                [raw_datasets[split] for split in raw_datasets.keys()]
            )
            raw_datasets = raw_datasets.train_test_split(
                test_size=0.2, seed=args.split_seed
            )
            logger.info(f"Re-split data with seed {args.split_seed}")

        # Truncate data if needed
        if args.proportion < 1:
            new_len = int(len(raw_datasets["train"]) * args.proportion)
        elif args.max_train_samples:
            new_len = args.max_train_samples
        else:
            new_len = None
        args.new_len = new_len

        # If debugging, only use a small subset of the data
        if args.debug:
            seeds = [args.seed]
            raw_datasets["train"] = raw_datasets["train"].select(range(100))
            raw_datasets["test"] = raw_datasets["test"].select(range(100))
            if "validation" in raw_datasets:
                raw_datasets["validation"] = raw_datasets["validation"].select(
                    range(min(100, len(raw_datasets["validation"])))
                )
            args.epochs = 1
            logger.info("Using a subset of the data for debugging")
        

        # Define dicts
        label2id = {}
        id2label = {}
        for i, label in enumerate(variables["LABELS"]):
            label2id[label] = i
            id2label[i] = label

        device = current_device() if is_available() else "cpu"
        logger.info(
            f"Main Training Arguments: device={device}, batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}"
        )

        # Labels to ids
        if "label" in raw_datasets["train"].features and raw_datasets["train"].features["label"].dtype == "string":
            raw_datasets = raw_datasets.map(lambda x: {"label": label2id[x["label"]]})
        


        if task == "trc":
            class_weights = calculate_class_weights(raw_datasets)

        if "validation" not in raw_datasets:
            # Set the validation set to the test set
            raw_datasets["validation"] = raw_datasets["test"]

        # Train models
        for model_info in models_info:
            # Clear flota flag
            args.flota = orig_args.flota

            logger.info(f"Training model from checkpoint {model_info['trained_ckpt']}")

            # Extract model architecture, should be one of ["bert", "roberta] for base config
            args.model_arc = "roberta" if "HeRo" in model_info["base_ckpt"] else "bert"
            
            if task == "trc":
                for arc in args.architectures:
                    __, f1_scores, model_name_results = _run_eval_wrapper(
                        model_info,
                        args,
                        logger,
                        seeds,
                        raw_datasets,
                        label2id,
                        id2label,
                        class_weights=class_weights,
                        compute_metrics=compute_metrics,
                        prepare_for_training=prepare_for_training,
                        evaluate=evaluate,
                        arc=arc,
                    )
                    # Add results to df
                    results_df.loc[model_name_results, f"{arc}_mean"] = round(
                        np.mean(f1_scores), 3
                    )
                    results_df.loc[model_name_results, f"{arc}_std"] = round(
                        np.std(f1_scores), 2
                    )
            else:
                raise ValueError(f"Task {task} not supported")

        # Set up the results summary path
        cur_res_summary_path = f"{args.summary_dir}/{task}_results"
        if args.probing:
            cur_res_summary_path += "_probing.csv"
        else:
            cur_res_summary_path += ".csv"

        # Save results of this run
        results_df.to_csv(cur_res_summary_path)

        # Update the existing results df
        if args.update_existing_results:
            if existing_results_df is not None:
                existing_results_df.update(results_df)
                existing_results_df = existing_results_df.combine_first(results_df)
            else:
                existing_results_df = results_df
            # Save all results df
            existing_results_df.to_csv(args.results_path)

        logger.info(f"Finished running {task} evaluation")
