import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from abc import ABC
from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from src.constants import *
from src.modules import GPTModule, PositionalEncoding


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
    if similarity == "l2":

        def h_fn(context_inputs, queries):
            return -jnp.sum((context_inputs - queries) ** 2, axis=-1, keepdims=True)

        return h_fn
    else:
        raise NotImplementedError


def make_g(ground_truth_prob: float):
    def g_fn(queries, outputs, flip_labels):
        outputs = flip_labels * (1 - outputs) + (1 - flip_labels) * outputs
        return jnp.clip(
            jnp.abs(jnp.full_like(
                outputs, fill_value=((max(ground_truth_prob, 1 - ground_truth_prob)) / (outputs.shape[-1] - 1))
            ) - outputs),
            a_min=0.0,
            a_max=1.0,
        )

    return g_fn


class SimpleICLModel(Model):
    def __init__(
        self,
        ground_truth_prob: float,
        similarity: str,
        temperature: float = 0.1,
        alpha_num_examples: int = 0,
    ):
        self.alpha = nn.Dense(1)
        self.h_fn = make_h(similarity)
        self.g_fn = make_g(ground_truth_prob)
        self.temperature = temperature
        self.alpha_num_examples = alpha_num_examples

        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        return {
            "alpha": self.alpha.init(
                model_key,
                np.array(
                    [input_space.sample()] * (self.alpha_num_examples + 1)
                ).flatten(),
            )
        }

    def make_forward(
        self,
        mutable,
        capture_intermediates=False,
    ):

        if self.alpha_num_examples > 0:

            def alpha_forward(params, batch):
                return self.alpha.apply(
                    params, batch["example"].reshape((len(batch["example"]), -1))
                )

        else:

            def alpha_forward(params, batch):
                return self.alpha.apply(params, batch["example"][:, -1])

        def forward(
            params,
            batch,
            eval=False,
            **kwargs,
        ):
            queries = batch["example"][:, -1]
            targets = batch["target"][:, -1]
            flip_labels = batch["flip_label"][:, None]
            context_inputs = batch["example"][:, :-1]
            context_targets = batch["target"][:, :-1]

            alphas = alpha_forward(params["alpha"], batch)
            p_iwl = jax.nn.sigmoid(alphas)
            similarity = self.h_fn(context_inputs, queries[:, None])
            ic_pred = jnp.sum(
                jax.nn.softmax(similarity / self.temperature, axis=1) * context_targets,
                axis=1,
            )
            iw_pred = self.g_fn(queries, targets, flip_labels)

            return (1 - p_iwl) * ic_pred + p_iwl * iw_pred, {
                "alpha": alphas,
                "p_iwl": p_iwl,
                "h": similarity,
                "iw_pred": iw_pred,
                "ic_pred": ic_pred,
            }

        return forward


class SimpleICLModelLearnedIWPredictor(Model):
    def __init__(
        self,
        similarity: str,
        temperature: float = 0.1,
        alpha_num_examples: int = 0,
    ):
        self.alpha = nn.Dense(1)
        self.h_fn = make_h(similarity)
        self.g_fn = nn.Dense(2)
        self.temperature = temperature
        self.alpha_num_examples = alpha_num_examples

        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        alpha_key, g_key = jrandom.split(model_key)
        sample = input_space.sample()
        return {
            "alpha": self.alpha.init(
                alpha_key,
                np.array(
                    [input_space.sample()] * (self.alpha_num_examples + 1)
                ).flatten(),
            ),
            "g": self.g_fn.init(g_key, sample),
        }

    def make_forward(
        self,
        mutable,
        capture_intermediates=False,
    ):

        if self.alpha_num_examples > 0:

            def alpha_forward(params, batch):
                return self.alpha.apply(
                    params, batch["example"].reshape((len(batch["example"]), -1))
                )

        else:

            def alpha_forward(params, batch):
                return self.alpha.apply(params, batch["example"][:, -1])

        def forward(
            params,
            batch,
            eval=False,
            **kwargs,
        ):
            queries = batch["example"][:, -1]
            targets = batch["target"][:, -1]
            flip_labels = batch["flip_label"][:, None]
            context_inputs = batch["example"][:, :-1]
            context_targets = batch["target"][:, :-1]

            alphas = alpha_forward(params["alpha"], batch)
            p_iwl = jax.nn.sigmoid(alphas)
            similarity = self.h_fn(context_inputs, queries[:, None])
            ic_pred = jnp.sum(
                jax.nn.softmax(similarity / self.temperature, axis=1) * context_targets,
                axis=1,
            )
            iw_pred = jax.nn.softmax(
                self.g_fn.apply(params["g"], queries) / self.temperature, axis=1
            )

            return (1 - p_iwl) * ic_pred + p_iwl * iw_pred, {
                "alpha": alphas,
                "p_iwl": p_iwl,
                "h": similarity,
                "iw_pred": iw_pred,
                "ic_pred": ic_pred,
            }

        return forward


class InContextSupervisedTransformer(Model):
    """A GPT for in-context learning."""

    def __init__(
        self,
        output_dim: int,
        num_contexts: int,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        widening_factor: int,
        query_pred_only: bool = False,
        input_output_same_encoding: bool = True,
        freeze_input_tokenizer: bool = True,
        **kwargs,
    ) -> None:
        self.gpt = GPTModule(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            widening_factor=widening_factor,
        )
        self.input_tokenizer = nn.Dense(embed_dim)
        self.output_tokenizer = nn.Dense(embed_dim)
        self.predictor = nn.Dense(int(np.product(output_dim)))
        self.num_tokens = num_contexts * 2 + 1
        self.positional_encoding = PositionalEncoding(
            embed_dim,
            self.num_tokens,
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.freeze_input_tokenizer = freeze_input_tokenizer
        self.apply_positional_encoding = self._make_get_positional_encoding(
            input_output_same_encoding
        )
        self.tokenize = jax.jit(self.make_tokenize())
        self.get_latent = jax.jit(self.make_get_latent())
        self.forward = jax.jit(self.make_forward(query_pred_only))

    def _make_get_positional_encoding(
        self, input_output_same_encoding: bool
    ) -> Callable:
        if input_output_same_encoding:

            def apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
            ):
                # Treat input-output pair with same position
                input_embedding = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    input_embedding,
                    **kwargs,
                )

                context_input_embedding, query_embedding = (
                    input_embedding[:, :-1],
                    input_embedding[:, -1:],
                )
                context_output_embedding = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    context_output_embedding,
                    **kwargs,
                )

                stacked_inputs = jnp.concatenate(
                    (context_input_embedding, context_output_embedding), axis=-1
                ).reshape((len(queries), -1, self.embed_dim))

                stacked_inputs = jnp.concatenate(
                    (stacked_inputs, query_embedding), axis=1
                )
                return stacked_inputs

        else:

            def apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
            ):
                # Treat each token separately position
                context_input_embedding, query_embedding = (
                    input_embedding[:, :-1],
                    input_embedding[:, -1:],
                )
                stacked_inputs = jnp.concatenate(
                    (
                        context_input_embedding,
                        context_output_embedding,
                    ),
                    axis=-1,
                ).reshape((len(queries), -1, self.embed_dim))
                stacked_inputs = jnp.concatenate(
                    (
                        stacked_inputs,
                        query_embedding,
                    ),
                    axis=1,
                )
                stacked_inputs = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    stacked_inputs,
                    **kwargs,
                )
                return stacked_inputs

        return apply_positional_encoding

    def init(
        self,
        model_key: jrandom.PRNGKey,
        input_space,
        output_space,
    ) -> Union[optax.Params, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param input_space: the input space
        :param output_space: the output space
        :type model_key: jrandom.PRNGKey
        :return: the initialized parameters
        :rtype: Union[optax.Params, Dict[str, Any]]

        """
        input_key, output_key, gpt_key, predictor_key, pe_key = jrandom.split(
            model_key, 5
        )
        dummy_token = np.zeros((1, 1, self.embed_dim))
        dummy_repr = np.zeros((1, 1, self.embed_dim))
        return {
            CONST_INPUT_TOKENIZER: (
                {
                    "params": {
                        "kernel": jnp.eye(self.embed_dim),
                        "bias": jnp.zeros(self.embed_dim),
                    }
                }
                if self.freeze_input_tokenizer
                else self.input_tokenizer.init(input_key, input_space.sample()[None])
            ),
            CONST_OUTPUT_TOKENIZER: self.output_tokenizer.init(
                output_key, np.zeros(output_space.n)[None]
            ),
            CONST_GPT: self.gpt.init(gpt_key, dummy_token, eval=True),
            CONST_POSITIONAL_ENCODING: self.positional_encoding.init(
                pe_key, dummy_token
            ),
            CONST_PREDICTOR: self.predictor.init(predictor_key, dummy_repr),
        }

    def make_tokenize(
        self,
    ) -> Callable:
        """
        Makes the tokenize call of the ICL model.

        :return: the tokenize call.
        :rtype: Callable
        """

        def tokenize(
            params: Union[optax.Params, Dict[str, Any]],
            batch: Any,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Get latent call of the GPT.

            :param params: the model parameters
            :param batch: the batch
            :type params: Union[optax.Params, Dict[str, Any]]
            :type batch: Any
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """

            input_embedding, input_updates = self.input_tokenizer.apply(
                params[CONST_INPUT_TOKENIZER],
                batch["example"],
                mutable=[CONST_BATCH_STATS],
            )

            context_output_embedding, output_updates = self.output_tokenizer.apply(
                params[CONST_OUTPUT_TOKENIZER],
                batch["target"][:, :-1],
                mutable=[CONST_BATCH_STATS],
            )

            stacked_inputs = self.apply_positional_encoding(
                params,
                batch["example"][:, -1],
                input_embedding,
                context_output_embedding,
                **kwargs,
            )

            return (
                stacked_inputs,
                {
                    CONST_INPUT_TOKENIZER: input_updates,
                    CONST_OUTPUT_TOKENIZER: output_updates,
                },
            )

        return tokenize

    def make_get_latent(
        self,
    ) -> Callable:
        """
        Makes the get latent call of the ICL model.

        :return: the get latent call.
        :rtype: Callable
        """

        def get_latent(
            params: Union[optax.Params, Dict[str, Any]],
            batch: Any,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            """
            Get latent call of the GPT.

            :param params: the model parameters
            :param batch: the batch
            :type params: Union[optax.Params, Dict[str, Any]]
            :type batch: Any
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            stacked_inputs, token_updates = self.tokenize(
                params, batch, eval, **kwargs
            )
            (repr, gpt_updates) = self.gpt.apply(
                params[CONST_GPT],
                stacked_inputs,
                eval,
                mutable=[CONST_BATCH_STATS],
                **kwargs,
            )

            return repr, {**token_updates, CONST_GPT: gpt_updates}

        return get_latent

    def make_forward(self, query_pred_only: bool) -> Callable:
        """
        Makes the forward call of the ICL model.

        :param query_pred_only: whether or not to output the query prediciton only
        :type query_pred_only: bool
        :return: the forward call.
        :rtype: Callable
        """

        if query_pred_only:

            def process_prediction(preds):
                return preds[:, -1]

        else:

            def process_prediction(preds):
                return preds[:, ::2]

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            batch: Any,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            """
            Forward call of the GPT.

            :param params: the model parameters
            :param batch: the batch
            :type params: Union[optax.Params, Dict[str, Any]]
            :type batch: Any
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            repr, latent_updates = self.get_latent(params, batch, eval, **kwargs)
            outputs = self.predictor.apply(
                params[CONST_PREDICTOR],
                repr,
            )

            return process_prediction(outputs), latent_updates

        return forward

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Any
    ) -> Dict[str, Any]:
        params[CONST_INPUT_TOKENIZER] = self.input_tokenizer.update_batch_stats(
            params[CONST_INPUT_TOKENIZER],
            batch_stats[CONST_INPUT_TOKENIZER],
        )
        params[CONST_OUTPUT_TOKENIZER] = self.output_tokenizer.update_batch_stats(
            params[CONST_OUTPUT_TOKENIZER],
            batch_stats[CONST_OUTPUT_TOKENIZER],
        )
        params[CONST_GPT] = self.output_tokenizer.update_batch_stats(
            params[CONST_GPT],
            batch_stats[CONST_GPT],
        )
        return params
