# In-context Learning

## Installation
Python 3.10
```
pip install tensorflow-cpu tensorflow-datasets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jax==0.4.30 # Or pip install -U "jax[cuda12]==0.4.30"
pip install flax==0.8.5 orbax-checkpoint==0.4.3
pip install chex optax dill gymnasium scikit-learn matplotlib seaborn
pip install prefetch_generator
```

## Example
```
python src/main.py --config_path=experiments/config/simple_icl_prob_1.0.json
```
