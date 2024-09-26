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
    os.makedirs(os.path.join(RUN_REPORT_DIR, "simulate_minibatch"), exist_ok=True)
    result_dir = os.path.join(EVAL_DIR, exp_name)
    num_runs = 0
    dat_content = ""

    for variant in os.listdir(result_dir):
        evaluation_file = os.path.join(result_dir, variant)

        if os.path.isdir(evaluation_file):
            continue

        num_runs += 1
        dat_content += "export evaluation_file={} \n".format(
            evaluation_file,
        )

    with open(os.path.join(CONFIG_DIR, "simulate_minibatch-{}.dat".format(exp_name)), "w+") as f:
        f.writelines(dat_content)

    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
    sbatch_content += "#SBATCH --time=01:00:00\n"

    if exp_name.startswith("omniglot"):
        sbatch_content += "#SBATCH --cpus-per-task=6\n"
        sbatch_content += "#SBATCH --mem=12G\n"
    else:
        sbatch_content += "#SBATCH --cpus-per-task=1\n"
        sbatch_content += "#SBATCH --mem=3G\n"

    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output={}/%j.out\n".format(
        os.path.join(RUN_REPORT_DIR, "simulate_minibatch", exp_name)
    )

    sbatch_content += "module load StdEnv/2020\n"
    sbatch_content += "module load python/3.10\n"
    sbatch_content += "source ~/simple_icl/bin/activate\n"

    sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
    sbatch_content += " < {}`\n".format(
        os.path.join(CONFIG_DIR, "simulate_minibatch-{}.dat".format(exp_name))
    )
    sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
    sbatch_content += 'echo "Current working directory is `pwd`"\n'
    sbatch_content += 'echo "Running on hostname `hostname`"\n'
    sbatch_content += "echo ${evaluation_file}\n"
    sbatch_content += 'echo "Starting run at: `date`"\n'

    if exp_name.startswith("omniglot"):
        sbatch_content += "tar xf $HOME/torch_datasets.tar -C $SLURM_TMPDIR\n"

    sbatch_content += "JAX_PLATFORMS=cpu python3 {}/plot_utils/simulate_minibatches.py \\\n".format(REPO_PATH)

    sbatch_content += "  --evaluation_file=${evaluation_file} \\\n"
    sbatch_content += "  --repo_path={} \\\n".format(REPO_PATH)
    sbatch_content += "  --results_dir={} \n".format(os.path.join(EVAL_DIR))
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    script_path = os.path.join(sbatch_dir, f"run_all-simulate_minibatch-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)

    run_all_content += "sbatch {}\n".format(script_path)

with open(
    "./sbatch_all_simulate_minibatch.sh",
    "w+",
) as f:
    f.writelines(run_all_content)
