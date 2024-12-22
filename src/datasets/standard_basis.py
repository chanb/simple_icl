import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np


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


        if self.conditioning == "none":
            basis = np.eye(self.num_dims)
            def create_examples(sample_rng):
                example_idxes = sample_rng.randint(self.num_dims, size=(self.num_contexts,))
                examples = basis[example_idxes]
                return examples
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

            copy_idx = sample_rng.randint(self.num_contexts)
            query = context_examples[copy_idx]
            target = context_labels[copy_idx]

            label = np.concatenate((context_labels, [target]))

            example = np.concatenate((context_examples, query[None]))
            one_hot = np.zeros((self.num_contexts + 1, 2))
            one_hot[np.arange(self.num_contexts + 1), label] = 1

            yield {
                "example": example,
                "label": one_hot,
            }
