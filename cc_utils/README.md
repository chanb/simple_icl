# Instructions

## Compute Canada Installation
```
module load python/3.10
module load StdEnv/2020

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
```

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
