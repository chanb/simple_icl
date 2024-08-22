"""
This script is the entrypoint for any experiment.
XXX: Try not to modify this.
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datetime import datetime
from pprint import pprint

import jax
import json
import logging
import os
import timeit
import uuid

from src.constants import *
from src.train import train
from src.utils import flatten_dict, parse_dict, set_seed, get_device


"""
This function constructs the model, optimizer, and learner and executes training.
"""


def main(config_path: str, run_seed: int = None, device: str = CONST_CPU):
    """
    Orchestrates the experiment.

    :param config_path: the experiment configuration file path
    :param run_seed: the seed to initialize the random number generators
    :param device: the JAX device to use, supports [`cpu`, `gpu:<device_ids>`]
    :type config_path: str
    :type run_seed: int: (Default value = None)
    :type device: str: (Default value = cpu)

    """
    tic = timeit.default_timer()

    get_device(device)
    set_seed(run_seed)
    assert os.path.isfile(config_path), f"{config_path} is not a file"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        hyperparameter_str = "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in dict(flatten_dict(config_dict)).items()
                ]
            )
        )
        config = parse_dict(config_dict)

    pprint(config)

    save_path = None
    if config.logging_config.save_path:
        optional_prefix = ""
        if config.logging_config.experiment_name:
            optional_prefix += f"{config.logging_config.experiment_name}-"
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        run_id = str(uuid.uuid4())
        save_path = os.path.join(
            config.logging_config.save_path, f"{optional_prefix}{time_tag}-{run_id}"
        )
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w+") as f:
            json.dump(config_dict, f)

    train(config, hyperparameter_str, save_path)
    toc = timeit.default_timer()
    print(f"Experiment Time: {toc - tic}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="Training configuration", required=True
    )
    parser.add_argument(
        "--run_seed", type=int, default=0, help="Seed for the run", required=False
    )
    parser.add_argument(
        "--device",
        type=str,
        default=CONST_CPU,
        help="JAX device to use. To specify specific GPU device, do gpu:<device_ids>",
        required=False,
    )
    args = parser.parse_args()

    config_path = args.config_path
    seed = args.run_seed
    device = args.device
    main(config_path, seed, device)
