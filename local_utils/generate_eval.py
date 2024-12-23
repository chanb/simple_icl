# FOR TRANSFORMER EVAL
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from local_utils.configs import EXPERIMENTS
from local_utils.constants import (
    CONFIG_DIR,
    EVAL_DIR,
    HOME_DIR,
    LOG_DIR,
    REPO_PATH,
)

NUM_GPUS = 0
AVAIL_GPUS = []
NUM_PARALLEL = NUM_GPUS if NUM_GPUS > 0 else 2

sbatch_dir = "./sbatch_scripts"
os.makedirs(sbatch_dir, exist_ok=True)

for exp_name, exp_config in EXPERIMENTS.items():
    result_dir = os.path.join(LOG_DIR, exp_name)
    num_runs = 0

    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "source {}/.venv/bin/activate\n".format(HOME_DIR)
    for variant in os.listdir(result_dir):
        learner_path = os.path.join(result_dir, variant)

        if "p_relevant_context_0.0" in variant or "p_relevant_context_1.0" in variant:
            continue

        num_runs += 1

        if exp_name.startswith("omniglot") and NUM_GPUS > 0:
            sbatch_content += "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python3 {}/experiments/evaluation.py \\\n".format(REPO_PATH)
            sbatch_content += "  --device=gpu:{} \\\n".format(AVAIL_GPUS[num_runs % NUM_GPUS])
        else:
            sbatch_content += "JAX_PLATFORMS=cpu python3 {}/experiments/evaluation.py \\\n".format(REPO_PATH)
        sbatch_content += "  --learner_path={} \\\n".format(learner_path)
        sbatch_content += "  --save_path={} &\n".format(os.path.join(EVAL_DIR, exp_name))

        if num_runs % NUM_PARALLEL == 0:
            sbatch_content += "wait\n"

    script_path = os.path.join(sbatch_dir, f"run_all-eval-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)
