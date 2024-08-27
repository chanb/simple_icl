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
from src.utils import parse_dict, load_config, iterate_models


def get_eval_datasets(
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
                config_dict["data_seed"] = test_data_seed

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
            for fixed_start_pos in range(0, context_len, 4):
                for flip_label in [False, True]:
                    start_pos_config_dict = copy.deepcopy(config_dict)
                    modify_seed(start_pos_config_dict)

                    dataset_kwargs = {
                        prob_key: 1,
                        "fixed_start_pos": fixed_start_pos,
                        "mode": "default",
                        "flip_label": flip_label,
                    }

                    start_pos_config_dict["dataset_kwargs"].update(dataset_kwargs)

                    start_pos_config = parse_dict(start_pos_config_dict)
                    configs[
                        "{}-{}-start_pos_{}{}".format(
                            split,
                            prob_key,
                            fixed_start_pos,
                            "-flip_label" if flip_label else "",
                        )
                    ] = start_pos_config

    return {
        eval_name: get_data_loader(config) for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    learner_path = args.learner_path
    batch_size = args.batch_size
    num_eval_samples = args.num_eval_samples
    test_data_seed = args.test_data_seed

    config_dict, config = load_config(learner_path)
    config_dict["batch_size"] = batch_size
    config = parse_dict(config_dict)

    context_len = config.dataset_kwargs.num_examples
    fixed_length = True

    datasets, dataset_configs = get_eval_datasets(
        config_dict,
        test_data_seed,
        context_len,
    )

    train_data_loader, train_dataset = get_data_loader(
        config,
    )
    datasets["pretraining"] = (
        train_data_loader,
        train_dataset,
    )
    dataset_configs["pretraining"] = config.dataset_kwargs

    prefetched_data = {}
    for eval_name in tqdm(datasets, postfix="Prefetching data"):
        data_loader, dataset = datasets[eval_name]
        data_iter = iter(data_loader)
        prefetched_data[eval_name] = dict(
            samples=[next(data_iter) for _ in range(num_eval_samples // batch_size)],
            dataset_output_dim=dataset.output_space.n,
        )

    try:
        stats = {eval_name: dict() for eval_name in datasets}
        checkpoint_steps = []
        for params, model, checkpoint_step in tqdm(iterate_models(learner_path)):
            checkpoint_steps.append(checkpoint_step)
            for eval_name in datasets:
                dataset, data_loader = datasets[eval_name]
                aux = evaluate(
                    model=model,
                    params=params,
                    prefetched_data=prefetched_data[eval_name],
                    max_label=None,
                    context_len=context_len,
                    fixed_length=fixed_length,
                )

                for aux_key in aux:
                    stats[eval_name].setdefault(aux_key, [])
                    stats[eval_name][aux_key].append(aux[aux_key])

        pickle.dump(
            {"checkpoint_steps": checkpoint_steps, "stats": stats},
            open(
                os.path.join(learner_path, "evaluation.pkl"),
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
        "--num_eval_samples",
        type=int,
        default=1000,
        help="The number of evaluation tasks",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size")
    parser.add_argument(
        "--test_data_seed",
        type=int,
        default=1000,
        help="The seed for generating the test data",
    )
    args = parser.parse_args()

    main(args)
