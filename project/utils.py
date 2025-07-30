"""
A module with utility functions to assist the experiment.
"""

from argparse import Namespace
import matplotlib.pyplot as plt

# Global variables
NAMES_DICT = {
    "bert": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "modernbert": "answerdotai/ModernBERT-base",
    "biobert": "dmis-lab/biobert-v1.1",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "scibert": "allenai/scibert_scivocab_uncased",
    "bluebert": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "hebert": "avichr/heBERT",
    "alephbert": "onlplab/alephbert-base",
    "abg": "imvladikon/alephbertgimmel-base-512",
    "hero": "HeNLP/HeRo",
    "longhero": "HeNLP/LongHeRo",
    "dictabert": "dicta-il/dictabert",
}

DA_SCHEMAS = ["simple", "ada_lm", "scratch"]

EMB_INIT_STRATEGIES = ["random", "zero", "avg"]

TASKS = ["trc"]

TRC_ARCHITECUTRES = ["ESS", "EMP", "SEQ_CLS"]


def extract_base_name(name: str) -> str:
    """
    Extract the base name of a path
    :param name: the name of the local model
    :return: the base name
    """
    if "alephbert" in name:
        return "alephbert"
    elif "hebert" in name:
        return "hebert"
    elif "abg" in name:
        return "abg"
    elif "hero" in name:
        if "long" in name:
            return "longhero"
        else:
            return "hero"
    elif "dictabert" in name:
        return "dictabert"
    elif "biolinkbert" in name:
        return "biolinkbert"
    elif name == "bert":
        return "bert"
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_model_name_or_path(model_name: str) -> str:
    """
    Get the full model name on HuggingFace model hub
    :param model_name: the short model name
    :return: the full model name
    """
    return NAMES_DICT[model_name]


def plot_graph(
    steps: list,
    values: list,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    show: bool = False,
):
    """
    Plot a graph
    :param steps: the steps of the graph
    :param values: the values of the graph
    :param title: the title of the graph
    :param xlabel: the label of the x axis
    :param ylabel: the label of the y axis
    :param save_path: the path to save the graph
    :param show: whether to show the graph
    """
    plt.figure()
    plt.plot(steps, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    if show:
        plt.show()


def initialize_models_info(names: list) -> list[dict[str, str]]:
    """
    Initialize the models info list
    :param names: the names of the models
    :return: a list of dictionaries with the model info
    """
    return [
        {
            "short_name": name,
            "base_ckpt": get_model_name_or_path(name),
            "tokenizer": None,
            "flota": True if "flota" in name else False,
        }
        for name in names
    ]


def validate_args(args: Namespace, type: str):
    """
    Validate the further pre-training configs
    :param args: the further pre-training configs
    :param type: the type of the configs
    """
    if type == "experiment":
        assert args.summary_dir, "No summary directory provided"
        assert not (
            args.do_da and args.do_eval and not args.do_pretraining
        ), "Cannot perform DA and TRC without further pretraining"
        assert args.do_da or args.do_pretraining or args.do_eval, "No task to perform"

    elif type == "pretraining":
        if args.model_base_names:
            for model_name in args.model_base_names:
                assert model_name in NAMES_DICT, f"Unknown model name: {model_name}"
        assert args.dataset_name, "No dataset name provided"
        assert args.output_dir, "No output directory provided"
        assert args.batch_size, "No batch size provided"
        assert args.learning_rate, "No learning rate provided"
        assert args.epochs, "No number of training epochs provided"
        assert 0 < args.proportion <= 1, "Invalid proportion of data provided"
        assert args.save_steps, "No save steps provided"

    elif type == "da":
        assert args.tokenizer_base_names, "No tokenizer base names provided"
        for tokenizer_name in args.tokenizer_base_names:
            assert (
                tokenizer_name in NAMES_DICT
            ), f"Unknown tokenizer name: {tokenizer_name}"
        assert args.corpus_name, "No corpus name provided"
        assert args.schemas, "No schemas provided"
        for schema in args.schemas:
            assert schema in DA_SCHEMAS, f"Unknown schema: {schema}"
        if "scratch" in args.schemas:
            assert len(args.schemas) == 1, "Cannot use scratch with other schemas"
        assert args.output_dir, "No output directory provided"
        assert args.new_emb_init, "No new embedding initialization strategy provided"
        assert (
            args.new_emb_init in EMB_INIT_STRATEGIES
        ), f"Unknown embedding initialization strategy: {args.new_emb_init}"
        if "simple" in args.schemas:
            assert args.simple_vocab_size, "No simple vocab size provided"
        if "ada_lm" in args.schemas:
            assert args.ada_lm_interval, "No ada_lm interval provided"
            assert args.ada_lm_th, "No ada_lm threshold provided"
        if args.do_idf:
            assert (
                args.idf_vocab_size or args.idf_max_score or args.idf_count_th>=0
            ), "No IDF parameters provided"
    elif type == "evaluation":
        if args.evaluate_local_models:
            assert args.local_models_path, "No local models path provided"
        for model_name in args.hf_models_to_train:
            assert model_name in NAMES_DICT, f"Unknown model name: {model_name}"
        assert args.num_seeds, "No number of seeds provided"
        assert args.eval_tasks, "No evaluation tasks provided"
        for task in args.eval_tasks:
            assert task in TASKS, f"Unknown task: {task}"
        assert args.proportion, "No proportion of data provided"
        assert args.batch_size, "No batch size provided"
        assert args.learning_rate, "No learning rate provided"
        assert args.epochs, "No number of training epochs provided"

        # TRC specific
        for arc in args.architectures:
            assert arc in TRC_ARCHITECUTRES, f"Unknown architecture: {arc}"

    else:
        raise ValueError(f"Unknown args type: {type}")
