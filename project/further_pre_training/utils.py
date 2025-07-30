"""
A module with utility functions the main function to run further pre-training.
"""

import os
import sys
from shutil import rmtree, copy
import logging
import subprocess
from argparse import Namespace
from typing import Callable
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _get_values_from_event_acc(event_acc: EventAccumulator, tag: str) -> list:
    """
    Get the steps and values of a specific tag from the tensorboard event accumulator
    :param event_acc: the tensorboard event accumulator
    :param tag: the tag of the values
    :return: a list of steps, a list of values
    """
    events = event_acc.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values


def _plot_summary(
    log_dir: str, summary_dir: str, plot_graph: Callable, show: bool = False
):
    """
    Plot the summary of the training process
    :param logging_dir: the tensorboard logging directory
    :param summary_dir: the directory to save the summary plot
    """
    # Load tensorboard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Read the training loss, eval loss and eval accuracy
    train_steps, train_loss_values = _get_values_from_event_acc(event_acc, "train/loss")
    eval_steps, eval_loss_values = _get_values_from_event_acc(event_acc, "eval/loss")
    _, eval_accuracy_values = _get_values_from_event_acc(event_acc, "eval/accuracy")

    # Plot and save the graphs
    plot_graph(
        train_steps,
        train_loss_values,
        "Training Loss",
        "Steps",
        "Loss",
        os.path.join(summary_dir, "train_loss.png"),
        show=show,
    )
    plot_graph(
        eval_steps,
        eval_loss_values,
        "Evaluation Loss",
        "Steps",
        "Loss",
        os.path.join(summary_dir, "eval_loss.png"),
        show=show,
    )
    plot_graph(
        eval_steps,
        eval_accuracy_values,
        "Evaluation Accuracy",
        "Steps",
        "Accuracy",
        os.path.join(summary_dir, "eval_accuracy.png"),
        show=show,
    )


def run_further_pretraining(
    args: Namespace, plot_graph: Callable, model_info: dict[str, str]
):
    """
    Run further pre-training
    :param args: the arguments for further pre-training
    :param plot_graph: the function to plot the summary graph
    :param: model_info: information about the current model
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Set train path
    data_dir = "further_pre_training/data"
    if not args.train_file:
        if args.dataset_name == "doctors":
            args.train_file = os.path.join(data_dir, "doctors.txt")
        elif args.dataset_name == "demo":
            args.train_file = os.path.join(data_dir, "demo.txt")
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    args.model_name_or_path = model_info["base_ckpt"]
    if model_info["tokenizer"]:
        MODEL_NAME = re.split(r"[\\/]", model_info["tokenizer"])[-1]
    else:
        MODEL_NAME = f"{args.dataset_name}_{model_info['short_name']}"
    
    

    # Add seed to model name
    MODEL_NAME += f"_seed{args.seed}"

    # Add heads randomization to model name if it is set
    if args.random_heads_ratio and args.random_heads_ratio > 0:
        MODEL_NAME += f"_rand_heads{args.random_heads_ratio}"

    # Add lr to model name if it is not 5e-5
    if args.learning_rate != 5e-5:
        MODEL_NAME += f"_lr{args.learning_rate}"

    # Add flota to model name if it is set
    if args.flota:
        MODEL_NAME += "_flota"

    # Add proportion to model name if it is not 1
    if args.proportion < 1:
        MODEL_NAME += f"_prop_{int(args.proportion * 100)}"
        logger.info(f"Training on {int(args.proportion * 100)}% of the data")

    # Add training data size to model name if it is set
    if args.train_data_size:
        MODEL_NAME += f"_samples_{args.train_data_size}"
        logger.info(f"Training on {args.train_data_size} samples")

    # Set up summary directory
    pre_training_summary_dir = os.path.join(args.summary_dir, MODEL_NAME)
    if not os.path.exists(pre_training_summary_dir):
        os.makedirs(pre_training_summary_dir)
    run_mlm_log = os.path.join(pre_training_summary_dir, "run_mlm.log")

    if args.debug:
        args.save_steps = 1
        args.logging_steps = 1

    # Set up output directory
    logger.info(f"Model name: {MODEL_NAME}")
    args.cur_output_dir = os.path.join(args.output_dir, MODEL_NAME)

    # Tensorboard logging directory
    args.log_dir = os.path.join(args.cur_output_dir, "tb_logs")
    if os.path.exists(args.log_dir) and args.overwrite_output_dir:
        logger.info(f"Removing existing logging directory: {args.log_dir}")
        rmtree(args.log_dir)

    COMMANDLINE = (
        f"python further_pre_training/run_mlm.py"
        f" --output_dir {args.cur_output_dir}"
        f" --train_file {args.train_file}"
        f" --validation_file {args.validation_file}"
        f" --num_train_epochs {args.epochs}"
        f" --logging_steps {args.logging_steps}"
        f" --logging_dir {args.log_dir}"
        f" --evaluation_strategy steps"
        f" --save_steps {args.save_steps}"
        f" --model_name_or_path {args.model_name_or_path}"
        f" --learning_rate {args.learning_rate}"
        f" --per_device_train_batch_size {args.batch_size}"
        f" --per_device_eval_batch_size {args.batch_size}"
        f" --do_train"
        f" --do_eval"
        f" --seed {args.seed}"
        f" --save_total_limit {args.save_total_limit}"
        f" --line_by_line"
        f" --proportion {args.proportion}"
    )

    # Add tokenizer path if it exists
    if model_info["tokenizer"]:
        COMMANDLINE += f" --tokenizer_name {model_info['tokenizer']}"
        COMMANDLINE += f" --new_emb_init {args.new_emb_init}"

    # Add boolean arguments to command line
    if args.overwrite_output_dir:
        COMMANDLINE += " --overwrite_output_dir"
    if args.flota:
        COMMANDLINE += " --flota"
    if args.streaming:
        COMMANDLINE += " --streaming"

    # If debugging, cut the number of training and validation samples
    if args.debug:
        COMMANDLINE += f" --max_steps {5}"
        COMMANDLINE += f" --max_eval_samples {10}"
        COMMANDLINE += f" --max_train_samples {20}"
    elif args.train_data_size:
        COMMANDLINE += f" --max_train_samples {args.train_data_size}"
    else:
        pass

    # If training from scratch
    if args.from_scratch:
        COMMANDLINE += " --from_scratch"
        args.random_heads_ratio = 1
        
    # If randomizing heads ratio
    if args.random_heads_ratio and args.random_heads_ratio > 0:
        COMMANDLINE += f" --random_heads_ratio {args.random_heads_ratio}"

    logger.info(f"Running command: {COMMANDLINE}")
    try:
        with open(run_mlm_log, "w") as f:
            _ = subprocess.run(COMMANDLINE, shell=True, check=True, stdout=f, stderr=f)
        logger.info("run_mlm.py finished successfully!")

    except subprocess.CalledProcessError as e:
        # Handle errors in called script
        logger.exception(
            f"run_mlm.py failed due to an error, exit code: {e.returncode}"
        )
        sys.exit(1)

    except Exception as e:
        # Handle other exceptions
        logger.exception(f"run_mlm.py failed with an unexpected error: {e}")

    logger.info("Summarizing Pretraining results...")
    # Copy all_results.json to summary directory
    copy(
        os.path.join(args.cur_output_dir, "all_results.json"),
        os.path.join(pre_training_summary_dir, "pre_training_results.json"),
    )
    # Plot summary
    _plot_summary(
        args.log_dir, pre_training_summary_dir, plot_graph, show=args.show_plots
    )

    logger.info("Plots saved to summary directory")

    # Return the trained model directory path
    return args.cur_output_dir, pre_training_summary_dir


class FlotaBERTTokenizer(object):
    """Runs Flota tokenization."""

    def __init__(self, tokenizer, k=10, strict=False):
        self.vocab = tokenizer.get_vocab()
        self.unk_token = tokenizer.unk_token
        self.special = "##"
        self.max_len = 18
        self.k = k
        self.strict = strict

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes a piece of text into its word pieces. This uses the Flota Segmentation algorithm to perform
        tokenization using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        text_split = re.findall(r"[\w]+|[^\s\w]", text)
        tokens = list()
        for w in text_split:
            tokens.extend(self.tokenize_word(w))
        return tokens

    def max_subword_split(self, w: str) -> tuple:
        """
        Split a word into the longest subwords that are in the vocabulary
        """
        # Run through all possible subword lengths
        for length in range(min(len(w), self.max_len), 0, -1):
            # Run through all possible subwords of the current length
            for i in range(0, len(w) - length + 1):
                if w[i] == "-":
                    continue
                subword = w[i : i + length]
                if i == 0:
                    if subword in self.vocab:
                        return subword, w[:i] + length * "-" + w[i + length :], i
                    elif not self.strict and self.special + subword in self.vocab:
                        return (
                            self.special + subword,
                            w[:i] + length * "-" + w[i + length :],
                            i,
                        )
                else:
                    if self.special + subword in self.vocab:
                        return (
                            self.special + subword,
                            w[:i] + length * "-" + w[i + length :],
                            i,
                        )
        return None, None, None

    def get_flota_dict(self, w: str, k: int) -> dict:
        """
        Get the dictionary of subwords for a word
        :param w: the word to split
        :param k: the maximum number of subwords to split the word into
        """
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if k == 1 or rest == len(rest) * "-":
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest, k - 1)
        flota_dict[i] = max_subword
        return flota_dict

    def tokenize_word(self, w: str) -> list[str]:
        if w in self.vocab:
            return [w]
        elif self.special + w in self.vocab:
            return [self.special + w]
        else:
            flota_dict = self.get_flota_dict(w, self.k)
            return [subword for _, subword in sorted(flota_dict.items())]
