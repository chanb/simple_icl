import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class StandardBasisSynthetic:
    def __init__(
        self,
        num_contexts: int,
        num_dims: int,
        seed: int,
        train: bool,
        conditioning: str = "none",
    ):
        self.num_dims = num_dims
        self.train = train
        self.seed = seed
        self.num_contexts = num_contexts
        self.conditioning = conditioning
        self.rng = np.random.RandomState(seed)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Discrete(2)

    def get_sequences(
        self,
    ):
        sample_rng = np.random.RandomState(self.rng.randint(0, 2**16) + int(self.train))

        basis = np.eye(self.num_dims)
        def create_examples(sample_rng):
            example_idxes = sample_rng.randint(self.num_dims, size=(self.num_contexts,))
            examples = basis[example_idxes]
            return examples

        if self.conditioning == "none":

            def get_query_target(context_examples, context_labels):
                copy_idx = sample_rng.randint(self.num_contexts)
                query = context_examples[copy_idx]
                target = context_labels[copy_idx]

                return query, target

        elif self.conditioning.startswith("ood_factor"):
            ood_factor = float(self.conditioning.split(":")[-1])

            def get_query_target(context_examples, context_labels):
                copy_idx = sample_rng.randint(self.num_contexts)
                query = context_examples[copy_idx] * ood_factor
                target = context_labels[copy_idx]

                return query, target

        elif self.conditioning.startswith("simplex"):
            num_bases = int(self.conditioning.split(":")[-1])

            def get_query_target(context_examples, context_labels):
                take_idxes = sample_rng.permutation(self.num_contexts)[:num_bases]

                alphas = sample_rng.randn(num_bases)
                alphas = softmax(alphas)

                query = np.sum(context_examples[take_idxes] * alphas[:, None], axis=0)
                query /= np.linalg.norm(query)

                one_hot = np.zeros((num_bases, 2))
                one_hot[np.arange(num_bases), context_labels[take_idxes]] = 1
                target = np.argmax(np.sum(one_hot * alphas[:, None], axis=0))

                return query, target

        else:
            raise NotImplementedError


        while True:
            task_vector = (
                sample_rng.randint(2, size=(self.num_dims,)) * 2 - 1
            ).astype(int)

            context_examples = create_examples(sample_rng)
            context_labels = (
                context_examples @ task_vector >= 0.0
            ).astype(int)

            query, target = get_query_target(context_examples, context_labels)

            label = np.concatenate((context_labels, [target]))

            example = np.concatenate((context_examples, query[None]))
            one_hot = np.zeros((self.num_contexts + 1, 2))
            one_hot[np.arange(self.num_contexts + 1), label] = 1

            yield {
                "example": example,
                "label": one_hot,
            }
