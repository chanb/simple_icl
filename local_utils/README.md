# Instructions

## Status
Noisy Label:
- All
  - Run `label_noise: 0.1`
- IW/IC/Alpha experiment
  - `p_relevant_context: [0.1]` (done)
- Transformer experiment
  - `p_relevant_context: [0.75, 0.99]` (running)
- Transformer context length experiment
  - `context_len: [2, 4, 8]`
Noisy Input:
- Transformer experiment
  - Do we still need more to show results?
- Omniglot
  - Still need to design what to run

## Installation
### CPU
```
python -m venv ~/simple_icl
source ~/simple_icl/bin/activate

pip install jax --no-index
pip install optax flax --no-index
pip install chex dill matplotlib tensorboard seaborn --no-index
pip install gymnasium --no-index
pip install torch torchvision --no-index
pip install tensorflow --no-index
pip install tensorflow_datasets --no-index
pip install scikit-learn --no-index
pip install prefetch_generator --no-index
```

## Setup JupyterHub
See [here](https://docs.alliancecan.ca/wiki/Advanced_Jupyter_configuration#Python_kernel)

## Running Experiments
**Important:** Modify `constants.py` to setup the correct credentials and paths.

1. Generate the slurm scripts using `generate_train.py`:
```
python generate_train.py
```
For each experiment `EXP_NAME` defined in `configs.py`, `generate_train.py` will generate a corresponding experiment file that is located in `CONFIG_DIR` (e.g. `<CONFIG_DIR>/<EXP_NAME>.dat`), and a corresponding bash script `./sbatch_scripts/run_all-<EXP_NAME>.sh`.
`generate_train.py` will also generate a bash script `sbatch_all_train.sh` that will kick off all experiments.

1. Kick off experiments using `sbatch_all_train.sh`:
```
chmod +x sbatch_all_train.sh
./sbatch_all_train.sh
```

## Evaluating Results
