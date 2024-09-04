import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json

from itertools import product

from cc_utils.configs import EXPERIMENTS
from cc_utils.constants import (
    CONFIG_DIR,
    HOME_DIR,
    LOG_DIR,
    RUN_REPORT_DIR,
    REPO_PATH,
    CC_ACCOUNT,
)

sbatch_dir = "./sbatch_scripts"
os.makedirs(sbatch_dir, exist_ok=True)


def set_dict_value(d, key, val):
    if key in d:
        d[key] = val

    for k in d:
        if isinstance(d[k], dict):
            set_dict_value(d[k], key, val)


run_all_content = "#!/bin/bash\n"
for exp_name, exp_config in EXPERIMENTS.items():
    os.makedirs(
        os.path.join(RUN_REPORT_DIR, exp_name),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(CONFIG_DIR, exp_name),
        exist_ok=True,
    )

    template_path = os.path.join(
        REPO_PATH, "cc_utils/templates/{}.json".format(exp_name)
    )

    num_runs = 0
    dat_content = ""

    variant_keys = []
    variant_values = []
    for variant in exp_config["variants"]:
        variant_keys.append(variant["key"])
        variant_values.append(variant["values"])

    for seed in range(exp_config["num_seeds"]):
        for variant_config in product(*variant_values):
            variant_name = "-".join([
                "{}_{}".format(variant_key, variant_value) for variant_key, variant_value in zip(variant_keys, variant_config)
            ] + ["seed_{}".format(seed)])

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

            json.dump(
                config_dict,
                open(curr_config_path, "w"),
            )

            num_runs += 1
            dat_content += "export config_path={} \n".format(
                curr_config_path,
            )

    with open(os.path.join(CONFIG_DIR, "{}.dat".format(exp_name)), "w+") as f:
        f.writelines(dat_content)

    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
    sbatch_content += "#SBATCH --time={}\n".format(exp_config["run_time"])

    if exp_name.startswith("omniglot"):
        sbatch_content += "#SBATCH --cpus-per-task=4\n"
        sbatch_content += "#SBATCH --gres=gpu:1\n"
        sbatch_content += "#SBATCH --mem=6G\n"
    else:
        sbatch_content += "#SBATCH --cpus-per-task=1\n"
        sbatch_content += "#SBATCH --mem=3G\n"

    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output={}/%j.out\n".format(
        os.path.join(RUN_REPORT_DIR, exp_name)
    )

    if exp_name.startswith("omniglot"):
        sbatch_content += "module load StdEnv/2023\n"
        sbatch_content += "module load python/3.10\n"
        sbatch_content += "module load cuda/12.2\n"
        sbatch_content += "source ~/simple_icl_gpu/bin/activate\n"
    else:
        sbatch_content += "module load StdEnv/2020\n"
        sbatch_content += "module load python/3.10\n"
        sbatch_content += "source ~/simple_icl/bin/activate\n"

    sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
    sbatch_content += " < {}`\n".format(
        os.path.join(CONFIG_DIR, "{}.dat".format(exp_name))
    )
    sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
    sbatch_content += 'echo "Current working directory is `pwd`"\n'
    sbatch_content += 'echo "Running on hostname `hostname`"\n'
    sbatch_content += "echo ${config_path}\n"
    sbatch_content += 'echo "Starting run at: `date`"\n'

    if exp_name.startswith("omniglot"):
        sbatch_content += 'mkdir $SLURM_TMPDIR/tensorflow_datasets\n'
        sbatch_content += 'tar xf {} -C $SLURM_TMPDIR/tensorflow_datasets\n'.format(
            os.path.join(HOME_DIR, "tensorflow_datasets")
        )

    sbatch_content += "python3 {}/src/main.py \\\n".format(REPO_PATH)
    sbatch_content += "  --config_path=${config_path} \n"
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    script_path = os.path.join(sbatch_dir, f"run_all-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)

    run_all_content += "sbatch {}\n".format(script_path)

with open(
    "./sbatch_all_train.sh",
    "w+",
) as f:
    f.writelines(run_all_content)
