import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cc_utils.configs import EXPERIMENTS
from cc_utils.constants import (
    CONFIG_DIR,
    EVAL_DIR,
    HOME_DIR,
    LOG_DIR,
    RUN_REPORT_DIR,
    REPO_PATH,
    CC_ACCOUNT,
)

sbatch_dir = "./sbatch_scripts"
os.makedirs(sbatch_dir, exist_ok=True)

run_all_content = "#!/bin/bash\n"
for exp_name, exp_config in EXPERIMENTS.items():
    os.makedirs(os.path.join(RUN_REPORT_DIR, "eval"), exist_ok=True)
    result_dir = os.path.join(LOG_DIR, exp_name)
    num_runs = 0
    dat_content = ""

    for variant in os.listdir(result_dir):
        learner_path = os.path.join(result_dir, variant)

        if "p_relevant_context_0.0" in variant or "p_relevant_context_1.0" in variant:
            continue

        num_runs += 1
        dat_content += "export learner_path={} model_type={} \n".format(
            learner_path,
            "alpha"
        )
        num_runs += 1
        dat_content += "export learner_path={} model_type={} \n".format(
            learner_path,
            "ic"
        )
        num_runs += 1
        dat_content += "export learner_path={} model_type={} \n".format(
            learner_path,
            "iw"
        )

    with open(os.path.join(CONFIG_DIR, "eval-{}.dat".format(exp_name)), "w+") as f:
        f.writelines(dat_content)

    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
    sbatch_content += "#SBATCH --time={}\n".format(exp_config["eval_run_time"])

    if exp_name.startswith("omniglot"):
        sbatch_content += "#SBATCH --cpus-per-task=6\n"
        sbatch_content += "#SBATCH --gres=gpu:1\n"
        sbatch_content += "#SBATCH --mem=24G\n"
    else:
        sbatch_content += "#SBATCH --cpus-per-task=1\n"
        sbatch_content += "#SBATCH --mem=12G\n"

    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output={}/%j.out\n".format(
        os.path.join(RUN_REPORT_DIR, "eval", exp_name)
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
        os.path.join(CONFIG_DIR, "eval-{}.dat".format(exp_name))
    )
    sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
    sbatch_content += 'echo "Current working directory is `pwd`"\n'
    sbatch_content += 'echo "Running on hostname `hostname`"\n'
    sbatch_content += "echo ${learner_path} ${model_type}\n"
    sbatch_content += 'echo "Starting run at: `date`"\n'

    if exp_name.startswith("omniglot"):
        sbatch_content += "tar xf $HOME/torch_datasets.tar -C $SLURM_TMPDIR\n"
        sbatch_content += "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python3 {}/experiments/evaluation-e2e.py \\\n".format(REPO_PATH)
        sbatch_content += "  --device=gpu:0 \\\n"
    else:
        sbatch_content += "python3 {}/experiments/evaluation-e2e.py \\\n".format(REPO_PATH)

    sbatch_content += "  --learner_path=${learner_path} \\\n"
    sbatch_content += "  --model_type=${model_type} \\\n"
    sbatch_content += "  --save_path={} \n".format(os.path.join(EVAL_DIR, exp_name.replace("e2e_alpha", "${model_type}").replace("alpha", "${model_type}")))
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    script_path = os.path.join(sbatch_dir, f"run_all-eval-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)

    run_all_content += "sbatch {}\n".format(script_path)

with open(
    "./sbatch_all_eval.sh",
    "w+",
) as f:
    f.writelines(run_all_content)
