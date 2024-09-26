import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../src")

from experiments.utils import *

from src.constants import *
from src.dataset import get_data_loader
from src.utils import parse_dict, load_config, iterate_models, set_seed

from tqdm import tqdm

import _pickle as pickle
import argparse
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
import seaborn as sns
import pandas as pd
import timeit


parser = argparse.ArgumentParser()
parser.add_argument("--evaluation_file", type=str, required=True)
parser.add_argument("--repo_path", type=str, required=True)
parser.add_argument("--results_dir", type=str, required=True)
args = parser.parse_args()

evaluation_file = args.evaluation_file
repo_path = args.repo_path
results_dir = args.results_dir

variant_name = os.path.basename(os.path.dirname(evaluation_file))
run_name = os.path.basename(evaluation_file)

templates_dir = os.path.join(repo_path, "cc_utils", "templates")
out_dir = os.path.join(os.path.dirname(results_dir), "training_info", variant_name)

assert os.path.dirname(evaluation_file) != out_dir

os.makedirs(out_dir, exist_ok=True)

template_path = os.path.join(templates_dir, "{}.json".format(variant_name))
config_dict = json.load(open(template_path))

hyperparameters = run_name.split("-")[:-8]
for key_val_pair in hyperparameters:
    key = "_".join(key_val_pair.split("_")[:-1])
    value = key_val_pair.split("_")[-1]

    if key == "seed":
        config_dict["seeds"]["data_seed"] = int(value)
    else:
        config_dict["dataset_kwargs"][key] = int(value) if value.isdigit() else float(value)

config = parse_dict(config_dict)
loader, dataset = get_data_loader(config)

batch_size = config.batch_size
checkpoint_interval = config.logging_config.checkpoint_interval
num_epochs = config.num_epochs
num_high_freq_class = config.dataset_kwargs.num_high_prob_classes

batches = []
for epoch_i in tqdm(range(num_epochs)):
    batch = next(loader)
    if (epoch_i + 1) % checkpoint_interval == 0:
        target = batch["target"]
        labels = np.argmax(target, axis=-1)

        relevant_contexts = np.sum(labels[:, :-1] == labels[:, [-1]], axis=-1) > 0
        num_relevant_contexts = relevant_contexts = np.sum(labels[:, :-1] == labels[:, [-1]], axis=-1)
        high_freq_classes = labels[:, -1] <= num_high_freq_class
        batches.append(dict(
            num_relevant_contexts=num_relevant_contexts,
            high_freq_classes=high_freq_classes,
        ))

pickle.dump(
    batches,
    open(
        os.path.join(out_dir, run_name),
        "wb"
    )
)
