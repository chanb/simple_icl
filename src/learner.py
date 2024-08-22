import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from types import SimpleNamespace
from typing import Any, Dict, Union, Sequence

import chex
import flax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import timeit

import src.models as models

from src.constants import *
from src.dataset import get_data_loader
from src.optimizer import get_optimizer


def l2_norm(params: chex.PyTreeDef) -> chex.Array:
    """
    Computes the L2 norm of a complete PyTree.

    :param params: the pytree object with scalars
    :type params: PyTreeDef
    :return: L2 norm of the complete PyTree
    :rtype: chex.Array

    """
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))


def gather_learning_rate(
    aux: Dict,
    model_name: str,
    opt_state_list: Sequence[Any],
):
    for opt_state in opt_state_list:
        hyperparams = getattr(opt_state, CONST_HYPERPARAMS, {})
        if CONST_LEARNING_RATE in hyperparams:
            aux[f"{CONST_LEARNING_RATE}/{model_name}"] = hyperparams[
                CONST_LEARNING_RATE
            ].item()


class InContextLearner:
    """
    In-context learner.
    """

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        self._config = config
        self._num_updates_per_epoch = config.num_updates_per_epoch
        self._learner_key = jrandom.PRNGKey(config.learner_seed)

        self._train_data_loader, self._dataset = get_data_loader(
            config,
        )
        self._train_loader = iter(self._train_data_loader)

        self._initialize_model_and_opt()
        self._initialize_losses()
        self.train_step = jax.jit(self.make_train_step())

    def close(self):
        del self._train_loader
        del self._train_data_loader

    @property
    def model(self):
        """
        Model
        """
        return self._model

    @property
    def model_dict(self):
        """
        Model states
        """
        return self._model_dict

    def _initialize_model_and_opt(self):
        """
        Construct the model and the optimizer.
        """
        self._model = getattr(models, self._config.model_config.architecture)(
            **vars(self._config.model_config.model_kwargs)
        )

        model_key = jrandom.split(self._learner_key)[0]

        restore_path = getattr(self._config.model_config, "restore_path", False)
        if restore_path:
            import dill

            self._model_dict = dill.load(open(restore_path, "rb"))[CONST_MODEL_DICT]
            self._optimizer = get_optimizer(
                self._config.optimizer_config, self._model_dict[CONST_MODEL]
            )
        else:
            params = self._model.init(
                model_key, self._dataset.input_space, self._dataset.output_space
            )
            self._optimizer = get_optimizer(self._config.optimizer_config, params)
            opt_state = self._optimizer.init(params)
            self._model_dict = {CONST_MODEL: params, CONST_OPT_STATE: opt_state}

    def _initialize_losses(self):
        def cross_entropy(params, batch):
            logits, updates = self.model.forward(
                params,
                batch,
            )

            targets = batch["target"][:, -1]

            loss = jnp.mean(optax.softmax_cross_entropy(logits, targets))

            return loss, {
                CONST_UPDATES: updates,
            }

        self._loss = jax.jit(cross_entropy)

    def make_train_step(self):
        """
        Makes the training step for model update.
        """

        def _train_step(
            model_dict: Dict[str, Any],
            batch,
            *args,
            **kwargs,
        ) -> Any:
            """
            The training step that computes the loss and performs model update.

            :param model_dict: the model state and optimizer state
            :param batch: the samples
            :type model_dict: Dict[str, Any]
            :type batch: Dict[str, Any]
            :return: the updated model state and optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
            """
            (agg_loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(
                model_dict[CONST_MODEL],
                batch,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {CONST_MODEL: l2_norm(grads)}

            updates, opt_state = self._optimizer.update(
                grads,
                model_dict[CONST_OPT_STATE],
                model_dict[CONST_MODEL],
            )
            params = optax.apply_updates(model_dict[CONST_MODEL], updates)

            return {CONST_MODEL: params, CONST_OPT_STATE: opt_state}, aux

        return _train_step

    def update(self, epoch: int, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model.

        :param epoch: the epoch
        :type epoch: int
        :return: the update information
        :rtype: Dict[str, Any]

        """
        auxes = []
        total_sample_time = 0
        total_update_time = 0
        for update_i in range(self._num_updates_per_epoch):
            tic = timeit.default_timer()
            try:
                batch = next(self._train_loader)
            except StopIteration:
                self._train_loader = iter(self._train_dataloader)
                batch = next(self._train_loader)
            total_sample_time += timeit.default_timer() - tic

            self._learner_key = jrandom.fold_in(
                self._learner_key, epoch * self._num_updates_per_epoch + update_i
            )
            batch[CONST_RANDOM_KEY] = self._learner_key

            tic = timeit.default_timer()
            self._model_dict, aux = self.train_step(self._model_dict, batch)
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes.append(aux)

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        log = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AGG_LOSS].item(),
            f"time/{CONST_SAMPLE_TIME}": total_sample_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"{CONST_GRAD_NORM}/model": auxes[CONST_GRAD_NORM][CONST_MODEL].item(),
        }

        if isinstance(self._model_dict[CONST_OPT_STATE], dict):
            for model_name, opt_state_list in self._model_dict[CONST_OPT_STATE]:
                gather_learning_rate(aux, model_name, opt_state_list)
        else:
            gather_learning_rate(aux, CONST_MODEL, self._model_dict[CONST_OPT_STATE])

        return log
