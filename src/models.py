import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from abc import ABC
from flax import linen as nn
from typing import Any, Callable, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from src.constants import *


class Model(ABC):
    """Abstract model class."""

    #: Model forward call.
    forward: Callable

    #: Initialize model parameters.
    init: Callable

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        return params


def make_h(similarity: str):
    if similarity == "gaussian":

        def h_fn(context_inputs, queries):
            return jnp.exp(
                -jnp.sum((context_inputs - queries) ** 2, axis=-1, keepdims=True)
            )

        return h_fn
    else:
        raise NotImplementedError


def make_g(ground_truth_prob: float):
    def g_fn(queries, outputs):
        return jnp.clip(
            jnp.full_like(
                outputs, fill_value=((1 - ground_truth_prob) / (outputs.shape[-1] - 1))
            )
            + outputs,
            a_min=0.0,
            a_max=ground_truth_prob,
        )

    return g_fn


class SimpleICLModel(Model):
    def __init__(
        self,
        ground_truth_prob: float,
        similarity: str,
    ):
        self.alpha = nn.Dense(2)
        self.h_fn = make_h(similarity)
        self.g_fn = make_g(ground_truth_prob)
        
        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        return {"alpha": self.alpha.init(model_key, input_space.sample())}

    def make_forward(
        self,
        mutable,
        capture_intermediates=False,
    ):
        def forward(
            params,
            batch,
            eval=False,
            **kwargs,
        ):
            queries = batch["example"][:, -1]
            targets = batch["target"][:, -1]
            flip_labels = batch["flip_label"][:, None]
            targets = flip_labels * (1 - targets) + (1 - flip_labels) * targets
            context_inputs = batch["example"][:, :-1]
            context_targets = batch["target"][:, :-1]

            alphas = self.alpha.apply(params["alpha"], queries)
            p_iwl = jax.nn.softmax(alphas, axis=1)
            similarity = self.h_fn(context_inputs, queries[:, None])
            icl_pred = jnp.sum(
                jax.nn.softmax(similarity, axis=1) * context_targets,
                axis=1,
            )
            iwl_pred = self.g_fn(queries, targets)

            return (1 - p_iwl) * icl_pred + p_iwl * iwl_pred, {
                "alpha": alphas,
                "p_iwl": p_iwl,
                "h": similarity,
                "g": iwl_pred,
            }

        return forward
