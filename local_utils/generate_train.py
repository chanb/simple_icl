import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
import numpy as np

from itertools import product

from local_utils.configs import EXPERIMENTS
from local_utils.constants import (
    CONFIG_DIR,
    HOME_DIR,
    LOG_DIR,
    REPO_PATH,
)

NUM_GPUS = 1
AVAIL_GPUS = [1]
# NUM_PARALLEL = NUM_GPUS if NUM_GPUS > 0 else 10
NUM_PARALLEL = 2

sbatch_dir = "./sbatch_scripts"
os.makedirs(sbatch_dir, exist_ok=True)


def set_dict_value(d, key, val):
    if key in d:
        d[key] = val

    for k in d:
        if isinstance(d[k], dict):
            set_dict_value(d[k], key, val)

for exp_name, exp_config in EXPERIMENTS.items():
    os.makedirs(
        os.path.join(CONFIG_DIR, exp_name),
        exist_ok=True,
    )

    template_path = os.path.join(
        REPO_PATH, "local_utils/templates/{}.json".format(exp_name)
    )

    num_runs = 0

    if isinstance(exp_config["variants"][0]["key"], str):
        variant_keys = []
        variant_values = []
        for variant in exp_config["variants"]:
            variant_keys.append(variant["key"])
            variant_values.append(variant["values"])
        agg = product
    else:
        variant_keys = exp_config["variants"][0]["key"]
        variant_values = exp_config["variants"][0]["values"]
        agg = zip


    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "source {}/.venv/bin/activate\n".format(HOME_DIR)
    for seed in np.arange(exp_config["num_seeds"]) + 1:
        seed = int(seed)
        for variant_config in agg(*variant_values):
            variant_name = "-".join(
                [
                    ("{}_{}".format(variant_key, variant_value))
                    for variant_key, variant_value in zip(variant_keys, variant_config)
                ]
                + ["seed_{}".format(seed)]
            )
            print(variant_name)

            curr_config_path = os.path.join(
                CONFIG_DIR, exp_name, "{}.json".format(variant_name)
            )

            config_dict = json.load(open(template_path, "r"))

            config_dict["logging_config"]["save_path"] = os.path.join(LOG_DIR, exp_name)
            config_dict["logging_config"]["experiment_name"] = variant_name

            for seed_key in config_dict["seeds"]:
                config_dict["seeds"][seed_key] = seed

            for variant_key, variant_value in zip(variant_keys, variant_config):
                set_dict_value(config_dict, variant_key, variant_value)

            if exp_name == "synthetic-alpha":
                for (load_key, load_name) in (
                    ("load_iw", "synthetic-iw_predictor"),
                    ("load_ic", "synthetic-ic_predictor")
                ):
                    result_dir = os.path.join(LOG_DIR, load_name)
                    if not os.path.isdir(result_dir):
                        print("cannot find {} to load".format(load_name))
                        continue

                    for learner_path in os.listdir(result_dir):
                        if load_key == "load_iw":
                            include = True
                            for curr_attr in variant_name.split("-"):
                                if "p_relevant_context" in curr_attr:
                                    continue

                                if curr_attr not in learner_path:
                                    include = False
                                    break
                            if not include:
                                continue
                        else:
                            if not learner_path.startswith(variant_name):
                                continue

                        model_dir = os.path.join(
                            result_dir, learner_path, "models"
                        )
                        all_steps = [
                            filename
                            for filename in sorted(os.listdir(model_dir))
                        ]
                        config_dict["model_config"]["model_kwargs"][load_key] = os.path.join(
                            model_dir, "{}.dill".format(all_steps[-1])
                        )
                        break
                    else:
                        print("cannot find {} to load".format(load_name))

            json.dump(
                config_dict,
                open(curr_config_path, "w"),
            )

            num_runs += 1

            if exp_name.startswith("omniglot"):
                sbatch_content += "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python3 {}/src/main.py \\\n".format(REPO_PATH)
                sbatch_content += "  --device=gpu:{} \\\n".format(AVAIL_GPUS[num_runs % NUM_GPUS])
            else:
                sbatch_content += "python3 {}/src/main.py \\\n".format(REPO_PATH)
            sbatch_content += "  --config_path={} &\n".format(
                curr_config_path
            )

            if num_runs % NUM_PARALLEL == 0:
                sbatch_content += "wait\n"

    print("num_runs: {}".format(num_runs))
    script_path = os.path.join(sbatch_dir, f"run_all-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)
