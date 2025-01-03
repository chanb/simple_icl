import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
srcdir = os.path.join(os.path.dirname(parentdir), "src")
sys.path.insert(0, parentdir)
sys.path.insert(0, srcdir)

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, Any

import _pickle as pickle
import argparse
import copy
import timeit

from experiments.utils import *

from src.constants import *
from src.dataset import get_data_loader
from src.utils import parse_dict, load_config, iterate_models, set_seed, get_device


def get_omniglot_eval_datasets(
    config_dict: Dict[str, Any],
    test_data_seed: int,
    context_len: int,
    skip_test: bool = False,
):
    # In-weight
    in_weight_config_dict = copy.deepcopy(config_dict)
    in_weight_config_dict["dataset_kwargs"][
        "task_name"
    ] = "no_support"
    in_weight_config = parse_dict(in_weight_config_dict)

    # OOD N-shot 2-way
    test_n_shot_2_way_config_dict = copy.deepcopy(config_dict)
    test_n_shot_2_way_config_dict["dataset_kwargs"][
        "task_name"
    ] = "fewshot_holdout"
    test_n_shot_2_way_config_dict["dataset_kwargs"][
        "fs_shots"
    ] = 4
    test_n_shot_2_way_config = parse_dict(test_n_shot_2_way_config_dict)

    configs = {
        "in_weight": in_weight_config,
        "test_n_shot_2_way": test_n_shot_2_way_config,
    }

    return {
        eval_name: get_data_loader(
            config
        )
        for eval_name, config in configs.items()
    }, configs


def get_streamblock_eval_datasets(
    config_dict: Dict[str, Any],
    test_data_seed: int,
    context_len: int,
    skip_test: bool = False,
):
    configs = dict()
    for split in ["pretrain", "test"]:
        if skip_test and split == "test":
            continue

        if split == "test":

            def modify_seed(config_dict):
                config_dict["seeds"]["data_seed"] = test_data_seed

        else:

            def modify_seed(config_dict):
                pass

        icl_iid_context_config_dict = copy.deepcopy(config_dict)
        modify_seed(icl_iid_context_config_dict)
        dataset_kwargs = {"mode": "iid_context"}

        icl_iid_context_config_dict["dataset_kwargs"].update(dataset_kwargs)
        icl_iid_context = parse_dict(icl_iid_context_config_dict)

        configs[f"{split}-icl_iid_context"] = icl_iid_context

        # Context length evaluations
        for prob_key in ["sample_high_prob_class_only", "sample_low_prob_class_only"]:
            for context_key in ["sample_relevant_context", "sample_irrelevant_context"]:
                for flip_label in [False, True]:
                    start_pos_config_dict = copy.deepcopy(config_dict)
                    modify_seed(start_pos_config_dict)

                    dataset_kwargs = {
                        prob_key: 1,
                        context_key: 1,
                        "fixed_start_pos": 0 if context_key == "sample_irrelevant_context" else -1,
                        "mode": "default",
                        "flip_label": flip_label,
                    }

                    start_pos_config_dict["dataset_kwargs"].update(dataset_kwargs)

                    start_pos_config = parse_dict(start_pos_config_dict)
                    configs[
                        "{}-{}-{}{}".format(
                            split,
                            prob_key,
                            context_key,
                            "-flip_label" if flip_label else "",
                        )
                    ] = start_pos_config

    return {
        eval_name: get_data_loader(config) for eval_name, config in configs.items()
    }, configs


def get_synthetic_eval_datasets(
    config_dict: Dict[str, Any],
    test_data_seed: int,
    context_len: int,
    num_eval_samples: int,
    p_relevant_context: float = None,
    heldout: bool = False,
):
    configs = dict()

    if heldout:
        for relevant_context in ["default", "relevant_context", "irrelevant_context"]:
            for conditioning in ["none", "high_prob", "low_prob"]:
                eval_config_dict = copy.deepcopy(config_dict)

                eval_config_dict["dataset_kwargs"]["dataset_size"] = num_eval_samples * 5
                eval_config_dict["dataset_kwargs"]["train"] = False
                if relevant_context == "relevant_context":
                    eval_config_dict["dataset_kwargs"]["p_relevant_context"] = 1.0
                elif relevant_context == "irrelevant_context":
                    eval_config_dict["dataset_kwargs"]["p_relevant_context"] = 0.0
                elif p_relevant_context is not None:
                    eval_config_dict["dataset_kwargs"]["p_relevant_context"] = p_relevant_context
                eval_config_dict["dataset_kwargs"]["conditioning"] = conditioning
                eval_config_dict["dataset_kwargs"]["flip_label"] = 1
                eval_config_dict["dataset_kwargs"]["exemplar"] = "heldout"

                eval_config = parse_dict(eval_config_dict)
                configs["eval-{}-{}-heldout_input-flipped_label".format(relevant_context, conditioning)] = eval_config
    else:
        for flip_label in [1, 0]:
            for relevant_context in ["default", "relevant_context", "irrelevant_context"]:
                for conditioning in ["none", "high_prob", "low_prob"]:
                    eval_config_dict = copy.deepcopy(config_dict)

                    eval_config_dict["dataset_kwargs"]["dataset_size"] = num_eval_samples * 5
                    eval_config_dict["dataset_kwargs"]["train"] = False
                    if relevant_context == "relevant_context":
                        eval_config_dict["dataset_kwargs"]["p_relevant_context"] = 1.0
                    elif relevant_context == "irrelevant_context":
                        eval_config_dict["dataset_kwargs"]["p_relevant_context"] = 0.0
                    elif p_relevant_context is not None:
                        eval_config_dict["dataset_kwargs"]["p_relevant_context"] = p_relevant_context
                    eval_config_dict["dataset_kwargs"]["conditioning"] = conditioning
                    eval_config_dict["dataset_kwargs"]["flip_label"] = flip_label

                    eval_config = parse_dict(eval_config_dict)
                    configs["eval-{}-{}{}".format(relevant_context, conditioning, "-flip_label" if flip_label else "")] = eval_config

    return {
        eval_name: get_data_loader(config) for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    learner_path = args.learner_path
    save_path = args.save_path
    num_pretrain_samples = args.num_pretrain_samples
    batch_size = args.batch_size
    num_eval_samples = args.num_eval_samples
    test_data_seed = args.test_data_seed
    p_relevant_context = args.p_relevant_context
    num_workers = args.num_workers
    device = args.device
    model_type = args.model_type

    get_device(device)

    set_seed(0)

    os.makedirs(save_path, exist_ok=True)

    config_dict, config = load_config(learner_path)
    config_dict["batch_size"] = batch_size
    config_dict["num_workers"] = num_workers
    if "dataset_name" not in config_dict:
        config_dict["dataset_name"] = "streamblock"
    config = parse_dict(config_dict)

    if hasattr(config.model_config.model_kwargs, "num_contexts"):
        context_len = config.model_config.model_kwargs.num_contexts
    elif hasattr(config.dataset_kwargs, "num_examples"):
        context_len = config.dataset_kwargs.num_examples
    elif hasattr(config.dataset_kwargs, "num_contexts"):
        context_len = config.dataset_kwargs.num_contexts
    else:
        raise NotImplementedError

    fixed_length = True

    if config.dataset_name == "streamblock":
        datasets, dataset_configs = get_streamblock_eval_datasets(
            config_dict,
            test_data_seed,
            context_len,
        )
    elif config.dataset_name == "omniglot":
        datasets, dataset_configs = get_synthetic_eval_datasets(
            config_dict,
            test_data_seed,
            context_len,
            num_eval_samples,
            p_relevant_context,
            True,
        )
    elif config.dataset_name == "synthetic":
        datasets, dataset_configs = get_synthetic_eval_datasets(
            config_dict,
            test_data_seed,
            context_len,
            num_eval_samples,
            p_relevant_context,
            False,
        )

    train_ds, train_dataset = get_data_loader(
        config,
    )
    datasets["pretraining"] = (
        train_ds,
        train_dataset,
    )
    dataset_configs["pretraining"] = config.dataset_kwargs
    sample_key = jrandom.PRNGKey(config_dict["seeds"]["learner_seed"])

    prefetched_data = {}
    for eval_name in tqdm(datasets, postfix="Prefetching data"):
        ds, dataset = datasets[eval_name]
        prefetched_data[eval_name] = dict(
            samples=[next(ds) for _ in range((num_pretrain_samples if eval_name == "pretraining" else num_eval_samples) // batch_size)],
            dataset_output_dim=dataset.output_space.n,
        )

    try:
        stats = {eval_name: dict() for eval_name in datasets}
        checkpoint_steps = []
        for params, model, checkpoint_step in tqdm(iterate_models(learner_path)):
            checkpoint_steps.append(checkpoint_step)
            sample_key = jrandom.fold_in(sample_key, checkpoint_step)

            if model_type == "alpha":
                curr_model = model
                curr_params = params
            elif model_type == "iw":
                curr_model = model.iw_predictor
                curr_params = {
                    CONST_MODEL: params[CONST_MODEL]["iw_predictor"]
                }
            elif model_type == "ic":
                curr_model = model.ic_predictor
                curr_params = {
                    CONST_MODEL: params[CONST_MODEL]["ic_predictor"]
                }

            for eval_name in prefetched_data:
                aux = evaluate(
                    model=curr_model,
                    params=curr_params,
                    prefetched_data=prefetched_data[eval_name],
                    max_label=None,
                    sample_key=sample_key,
                    fixed_length=fixed_length,
                )
                sample_key = jrandom.split(sample_key)[0]

                for aux_key in aux:
                    stats[eval_name].setdefault(aux_key, [])
                    stats[eval_name][aux_key].append(aux[aux_key])

        pickle.dump(
            {"checkpoint_steps": checkpoint_steps, "stats": stats},
            open(
                os.path.join(save_path, "{}.pkl".format(os.path.basename(learner_path))),
                "wb",
            ),
        )
    finally:
        for eval_name in datasets:
            data_loader, dataset = datasets[eval_name]
            del data_loader
            del dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner_path",
        type=str,
        required=True,
        help="The experiment run to load from",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The directory to save to",
    )
    parser.add_argument(
        "--num_pretrain_samples",
        type=int,
        default=10000,
        help="The number of pretraining samples",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=1000,
        help="The number of evaluation samples",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Use particular simple ICL model"
    )
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size")
    parser.add_argument(
        "--test_data_seed",
        type=int,
        default=1000,
        help="The seed for generating the test data",
    )
    parser.add_argument(
        "--p_relevant_context",
        type=float,
        default=None,
        help="Probability of relevant contexts"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="The number of workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use",
    )
    args = parser.parse_args()

    main(args)
