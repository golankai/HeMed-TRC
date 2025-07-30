"""
A script to run an experiment.
Usage:
    python run_exp.py

Make sure to set the device to use in line 10 and all the configurations before:
    general configs in config.json
    further pre-training configs in further_pre_training/config.json
    TRC configs in trc/config.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json  # noqa: E402
import argparse  # noqa: E402
from argparse import Namespace  # noqa: E402
import logging  # noqa: E402
import traceback  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402

from transformers import set_seed  # noqa: E402

import utils  # noqa: E402
from further_pre_training.utils import run_further_pretraining  # noqa: E402
from evaluation.utils import run_eval
from domain_adaptation.run_da import run_domain_adaptation  # noqa: E402

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument(
    "--max_eval_train_samples",
    type=int,
    default=None,
    help="Max number of samples to use for training during evaluation"
)

def load_configs(path: str) -> Namespace:
    """
    Load defaults for training args.
    """
    with open(path, "r") as f:
        cfg_dict = json.load(f)

    return Namespace(**cfg_dict)


def set_up(args: Namespace) -> str:
    """
    Set up the experiment
    :param args: the experiment configs
    return: the path to the experiment log file
    """
    # Set up offline mode for hugging face
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Set up summary directory
    summary_dir = f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"
    args.summary_dir = os.path.join(args.summary_dir, summary_dir)
    os.makedirs(args.summary_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.summary_dir, "logs.log")
    print(f"Logging to {log_file}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    # Set up random seed
    set_seed(args.seed)


if __name__ == "__main__":
    # Load experiment configs
    args = load_configs("config.json")
    # Get CMD args
    cmd_args = parser.parse_args()
    # Update args with cmd args that are not None
    for key, value in vars(cmd_args).items():
        if value is not None or not hasattr(args, key):
            setattr(args, key, value)

    utils.validate_args(args, "experiment")

    # Set up summary directory and logging
    set_up(args)

    # Save the experiment config
    with open(os.path.join(args.summary_dir, "exp_config.json"), "w") as f:
        json.dump({"seed": args.seed}, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Running experiment! Do DA: {args.do_da}, Do further pre-training: {args.do_pretraining}, Do Eval: {args.do_eval}"
    )

    if args.do_da:
        # Perform domain adaptation
        # Load DA configs
        da_args = load_configs("domain_adaptation/config.json")
        utils.validate_args(da_args, "da")

        # Check if base tokenizers are valid
        assert (
            "hero" not in da_args.tokenizer_base_names
        ), "HeRo tokenizer is not supported for DA"
        # Save the DA config
        with open(
            os.path.join(args.summary_dir, "domain_adaptation_config.json"), "w"
        ) as f:
            json.dump(vars(da_args), f, indent=2)
        da_args = Namespace(**vars(args), **vars(da_args))

        logger.info(
            f"Start domain adaptation with {da_args.corpus_name} dataset. Schemas: {da_args.schemas}, base tokenizers: {da_args.tokenizer_base_names}"
        )

        # Set up the output directory
        da_args.output_dir = os.path.join(da_args.output_dir, da_args.corpus_name)
        if not os.path.exists(da_args.output_dir):
            os.makedirs(da_args.output_dir)

        # Run domain adaptation
        logger.info("Start Domain Adaptation")
        try:
            args.models_info = run_domain_adaptation(
                da_args, utils.get_model_name_or_path
            )
            logging.info("Finished Domain Adaptation")
        except Exception as e:
            logger.error(f"Error while running DA: {e}")
            logger.error(traceback.format_exc())

    else:
        args.models_info = None

    if args.do_pretraining:
        # Load further pre-training configs
        pretraining_args = load_configs("further_pre_training/config.json")
        utils.validate_args(pretraining_args, "pretraining")
        # Save the further pre-training config
        with open(
            os.path.join(args.summary_dir, "further_pre_training_config.json"), "w"
        ) as f:
            json.dump(vars(pretraining_args), f, indent=2)

        # Adapt the further pre-training configs from the experiment configs
        pretraining_args = Namespace(**vars(args), **vars(pretraining_args))

        # Set up the output directory
        pretraining_args.output_dir = os.path.join(
            pretraining_args.output_dir, pretraining_args.dataset_name
        )
        if not os.path.exists(pretraining_args.output_dir):
            os.makedirs(pretraining_args.output_dir)

        # If Domain Adaptation is done, continue with these models, otherwise set up the experiments to run
        if args.do_da:
            pretraining_args.new_emb_init = da_args.new_emb_init
            if "scratch" in da_args.schemas:
                pretraining_args.from_scratch = True
            else:
                pretraining_args.from_scratch = False
            logger.info(
                "Further pre-training will be done on the models (tokenizers) from the DA stage!"
            )
        else:
            logger.info(
                "Further pre-training will be done on the models from the correspondent config file!"
            )
            pretraining_args.from_scratch = False
            args.models_info = utils.initialize_models_info(
                pretraining_args.model_base_names,
            )

        # Run with each model and configuration
        for model_info in args.models_info:
            # Run further pre-training
            logger.info(f'Start further pre-training with {model_info["short_name"]}')
            trained_ckpt, pre_training_summary_dir = run_further_pretraining(
                pretraining_args, utils.plot_graph, model_info
            )
            model_info["trained_ckpt"] = trained_ckpt
            model_info["summary_dir"] = pre_training_summary_dir
            model_info["from_hf"] = False
            logger.info(
                f'Finished further pre-training with {model_info["short_name"]}'
            )
            logger.info(f"Trained model saved at {trained_ckpt}")

    if args.do_eval:
        # Load evaluation config
        eval_args = load_configs("evaluation/config.json")
        utils.validate_args(eval_args, "evaluation")

        if args.max_eval_train_samples is not None:
            eval_args.max_train_samples = args.max_eval_train_samples
        
        # Save the evaluation config
        with open(os.path.join(args.summary_dir, "evaluation_config.json"), "w") as f:
            json.dump(vars(eval_args), f, indent=2)

        # Adapt the evaluation configs from the experiment configs
        eval_args = Namespace(**vars(args), **vars(eval_args))




        # Define the checkpoints to use for evaluation
        if args.do_pretraining:
            # Directly use the trained model for evaluation
            logger.info(f"Evaluating the models after pretraining: {args.models_info}")
        elif args.do_da:
            # This is not a valid configuration
            logger.error(
                "Cannot perform DA and Evaluation without doing further pretraining in between!"
            )
            sys.exit(1)

        else:  # Evaluate trained models
            if eval_args.evaluate_local_models:  # Evaluate all local models
                plms = os.listdir(eval_args.local_models_path)
                plm_base_names = [utils.extract_base_name(plm) for plm in plms]
                args.models_info = utils.initialize_models_info(plm_base_names)

                for model_info, plm_path in zip(args.models_info, plms):
                    model_info["trained_ckpt"] = os.path.join(
                        eval_args.local_models_path, plm_path
                    )
                    model_info["from_hf"] = False

                logger.info(f"Evaluating local models from {eval_args.local_models_path}")

            if eval_args.evaluate_hf_models:  # Evaluate all HF models
                if args.models_info is None:
                    args.models_info = []
                hf_models_info = utils.initialize_models_info(
                    eval_args.hf_models_to_train
                )
                for model_info in hf_models_info:
                    model_info["trained_ckpt"] = model_info["base_ckpt"]
                    model_info["from_hf"] = True
                
                args.models_info.extend(hf_models_info)
                logger.info(
                    f"Evaluating the following HF checkpoints: {hf_models_info}"
                )

        assert len(args.models_info) > 0, "No checkpoints to evaluate"

        # Run evaluation
        try:
            run_eval(eval_args, args.models_info)
            logging.info("Finished evaluation")
        except Exception as e:
            logger.error(f"Error while running evaluation: {e}")
            logger.error(traceback.format_exc())
