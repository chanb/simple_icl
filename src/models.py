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
import dill
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from src.constants import *
from src.modules import (
    GPTModule,
    PositionalEncoding,
    ResNetV1Module,
    MLPModule,
    Temperature,
)


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


class IWPredictor(Model):
    def __init__(
        self,
        output_dim: int,
    ):
        self.iw_predictor = MLPModule(
            layers=[64, 64, output_dim],
            activation=nn.relu,
            output_activation=identity,
            use_batch_norm=False,
            use_bias=True,
            flatten=True,
        )

        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        iw_pred_key = jrandom.split(model_key)[0]
        return {
            "iw_predictor": self.iw_predictor.init(
                iw_pred_key,
                input_space.sample()[None],
                eval=True,
            ),
        }

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
            logits = self.iw_predictor.apply(
                params["iw_predictor"],
                queries,
                eval=eval,
            )

            return logits, {
                "probs": jax.nn.softmax(logits, axis=-1),
            }

        return forward


class ICPredictor(Model):
    def __init__(
        self,
        output_dim: int,
        similarity: str = "l2",
    ):
        self.similarity = make_h(similarity)
        self.temperature = Temperature()

        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        temp_key = jrandom.split(model_key)[0]
        return {
            "ic_predictor": self.temperature.init(
                temp_key,
            ),
        }

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
            context_inputs = batch["example"][:, :-1]
            context_targets = batch["target"][:, :-1]

            similarity = self.similarity(context_inputs, queries[:, None])
            temp = self.temperature.apply(params["ic_predictor"])
            ic_pred = jnp.sum(
                jax.nn.softmax(similarity / (jnp.exp(temp) + 1e-8), axis=1)
                * context_targets,
                axis=1,
            )
            log_probs = jnp.log(jnp.clip(ic_pred, a_min=1e-7))

            return log_probs, {
                "probs": ic_pred,
                "similarity": similarity,
            }

        return forward


class SimpleICL(Model):
    def __init__(
        self,
        output_dim: int,
        num_contexts: int = 0,
        similarity: str = "l2",
        load_iw: str = None,
        load_ic: str = None,
    ):
        self.alpha = MLPModule(
            layers=[64, 64, 1],
            activation=nn.relu,
            output_activation=identity,
            use_batch_norm=False,
            use_bias=True,
            flatten=True,
        )
        self.iw_predictor = IWPredictor(output_dim)
        self.ic_predictor = ICPredictor(output_dim, similarity)
        self.num_contexts = num_contexts
        self.load_iw = load_iw
        self.load_ic = load_ic

        self.forward = jax.jit(
            self.make_forward([CONST_BATCH_STATS]), static_argnames=[CONST_EVAL]
        )
        self.intermediates = jax.jit(
            self.make_forward([CONST_INTERMEDIATES], True), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key, input_space, output_space):
        alpha_key, ic_key, iw_key = jrandom.split(model_key, num=3)
        query = input_space.sample()

        if self.load_ic:
            ic_predictor_params = dill.load(open(self.load_ic, "rb"))[CONST_MODEL]
        else:
            ic_predictor_params = self.ic_predictor.init(
                ic_key, input_space, output_space
            )

        if self.load_iw:
            iw_predictor_params = dill.load(open(self.load_iw, "rb"))[CONST_MODEL]
        else:
            iw_predictor_params = self.iw_predictor.init(
                iw_key, input_space, output_space
            )

        return {
            "alpha": self.alpha.init(
                alpha_key,
                np.array([query] * (self.num_contexts + 1)).flatten()[None],
                eval=True,
            ),
            "iw_predictor": iw_predictor_params,
            "ic_predictor": ic_predictor_params,
        }

    def make_forward(
        self,
        mutable,
        capture_intermediates=False,
    ):

        if self.num_contexts > 0:

            def alpha_forward(params, batch):
                return self.alpha.apply(
                    params,
                    batch["example"].reshape((len(batch["example"]), -1)),
                    eval=False,
                )

        else:

            def alpha_forward(params, batch):
                return self.alpha.apply(params, batch["example"][:, -1], eval=False)

        def forward(
            params,
            batch,
            eval=False,
            **kwargs,
        ):
            alphas = alpha_forward(params["alpha"], batch)
            p_iwl = jax.nn.sigmoid(alphas)

            _, iw_updates = self.iw_predictor.forward(
                params["iw_predictor"],
                batch,
                eval=eval,
            )

            _, ic_updates = self.ic_predictor.forward(
                params["ic_predictor"],
                batch,
                eval=eval,
            )

            iw_pred = iw_updates["probs"]
            ic_pred = ic_updates["probs"]

            probs = jnp.clip((1 - p_iwl) * ic_pred + p_iwl * iw_pred, a_min=1e-8)
            log_probs = jnp.log(probs)

            return log_probs, {
                "alpha": alphas,
                "p_iwl": p_iwl,
                "iw_pred": iw_pred,
                "ic_pred": ic_pred,
            }

        return forward


def identity(x):
    return x


class InContextSupervisedGRU(Model):
    """A GRU for in-context learning."""

    def __init__(
        self,
        output_dim: int,
        embed_dim: int,
        input_tokenizer: str = "mlp",
        query_pred_only: bool = False,
        freeze_input_tokenizer: bool = True,
        **kwargs,
    ) -> None:
        self.gru = nn.RNN(nn.GRUCell(embed_dim))
        if input_tokenizer == "mlp":
            self.input_tokenizer = MLPModule(
                layers=[embed_dim],
                activation=identity,
                output_activation=identity,
                use_batch_norm=False,
                use_bias=True,
                flatten=False,
            )
        elif input_tokenizer == "resnet":
            self.input_tokenizer = ResNetV1Module(
                blocks_per_group=[2, 2, 2, 2],
                features=[12, 32, 32, embed_dim],
                stride=[1, 2, 2, 2],
                use_projection=[
                    True,
                    True,
                    True,
                    True,
                ],
                use_bottleneck=True,
                use_batch_norm=False,
            )
        else:
            raise NotImplementedError
        self.output_tokenizer = nn.Dense(embed_dim)
        self.predictor = nn.Dense(int(np.product(output_dim)))

        self.embed_dim = embed_dim
        self.freeze_input_tokenizer = freeze_input_tokenizer
        self.tokenize = jax.jit(self.make_tokenize())
        self.get_latent = jax.jit(self.make_get_latent())
        self.forward = jax.jit(self.make_forward(query_pred_only))

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
        input_key, output_key, gru_key, predictor_key = jrandom.split(
            model_key, 4
        )
        dummy_token = np.zeros((1, 1, self.embed_dim))
        dummy_repr = np.zeros((1, 1, self.embed_dim))

        return {
            CONST_INPUT_TOKENIZER: (
                {
                    "params": {
                        "Dense_0": {
                            "kernel": jnp.eye(self.embed_dim),
                            "bias": jnp.zeros(self.embed_dim),
                        }
                    }
                }
                if self.freeze_input_tokenizer
                else self.input_tokenizer.init(
                    input_key, input_space.sample()[None], eval=True
                )
            ),
            CONST_OUTPUT_TOKENIZER: self.output_tokenizer.init(
                output_key, np.zeros(output_space.n)[None]
            ),
            CONST_GRU: self.gru.init(gru_key, dummy_token),
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
                eval=eval,
                mutable=[CONST_BATCH_STATS],
            )

            context_output_embedding, output_updates = self.output_tokenizer.apply(
                params[CONST_OUTPUT_TOKENIZER],
                batch["target"][:, :-1],
                mutable=[CONST_BATCH_STATS],
            )

            stacked_inputs = jnp.concatenate(
                (input_embedding[:, :-1], context_output_embedding), axis=-1
            ).reshape((len(input_embedding), -1, self.embed_dim))

            stacked_inputs = jnp.concatenate(
                (stacked_inputs, input_embedding[:, [-1]]), axis=-1
            ).reshape((len(input_embedding), -1, self.embed_dim))

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
            Get latent call of the GRU.

            :param params: the model parameters
            :param batch: the batch
            :type params: Union[optax.Params, Dict[str, Any]]
            :type batch: Any
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            stacked_inputs, token_updates = self.tokenize(params, batch, eval, **kwargs)
            (repr, gru_updates) = self.gru.apply(
                params[CONST_GRU],
                stacked_inputs,
                **kwargs,
            )

            return repr, {**token_updates, CONST_GRU: gru_updates}

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
        return params


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
        input_tokenizer: str = "mlp",
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
        if input_tokenizer == "mlp":
            self.input_tokenizer = MLPModule(
                layers=[embed_dim],
                activation=identity,
                output_activation=identity,
                use_batch_norm=False,
                use_bias=True,
                flatten=False,
            )
        elif input_tokenizer == "resnet":
            self.input_tokenizer = ResNetV1Module(
                blocks_per_group=[2, 2, 2, 2],
                features=[12, 32, 32, embed_dim],
                stride=[1, 2, 2, 2],
                use_projection=[
                    True,
                    True,
                    True,
                    True,
                ],
                use_bottleneck=True,
                use_batch_norm=False,
            )
        else:
            raise NotImplementedError
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
        self.get_attention = jax.jit(
            self.make_get_attention(), static_argnames=[CONST_EVAL]
        )

    def make_get_attention(self):

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
            stacked_inputs, token_updates = self.tokenize(params, batch, eval, **kwargs)
            (repr, gpt_updates) = self.gpt.apply(
                params[CONST_GPT],
                stacked_inputs,
                eval,
                mutable=["intermediates"],
                capture_intermediates=True,
            )

            return repr, {**token_updates, CONST_GPT: gpt_updates}

        return get_latent

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
                        "Dense_0": {
                            "kernel": jnp.eye(self.embed_dim),
                            "bias": jnp.zeros(self.embed_dim),
                        }
                    }
                }
                if self.freeze_input_tokenizer
                else self.input_tokenizer.init(
                    input_key, input_space.sample()[None], eval=True
                )
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
                eval=eval,
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
            stacked_inputs, token_updates = self.tokenize(params, batch, eval, **kwargs)
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
        return params
