import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np


class BinarySynthetic:
    def __init__(
        self,
        dataset_size: int,
        num_contexts: int,
        num_high_freq_classes: int,
        num_low_freq_classes: int,
        p_balance: float,
        p_high: float,
        num_dims: int,
        seed: int,
        train: bool,
        conditioning: str = "none",
        input_noise_std: float = 0.0,
        label_noise: float = 0.0,
        num_relevant_contexts: int = None,
        flip_label: bool = False,
    ):
        assert 0.0 < p_high < 1.0
        assert 0.0 < p_balance < 1.0
        assert p_high / num_high_freq_classes >= (1 - p_high) / num_low_freq_classes
        assert num_relevant_contexts is None or num_relevant_contexts > 0

        self.num_high_freq_classes = num_high_freq_classes
        self.num_low_freq_classes = num_low_freq_classes
        self.num_classes = num_high_freq_classes + num_low_freq_classes
        self.p_high = p_high
        self.p_balance = p_balance
        self.p_low = 1 - p_high
        self.num_dims = num_dims
        self.dataset_size = dataset_size
        self.train = train
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.label_noise = label_noise
        self.num_contexts = num_contexts
        self.num_relevant_contexts = num_relevant_contexts
        self.conditioning = conditioning
        self.flip_label = flip_label
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
        weights = [
            self.p_high / self.num_high_freq_classes
        ] * self.num_high_freq_classes + [
            self.p_low / self.num_low_freq_classes
        ] * self.num_low_freq_classes
        weights_high_freq = [
            1 / self.num_high_freq_classes
        ] * self.num_high_freq_classes
        weights_low_freq = [1 / self.num_low_freq_classes] * self.num_low_freq_classes

        def create_example(seed):
            example = np.random.RandomState(seed).standard_normal(self.num_dims)
            example /= np.linalg.norm(example)
            return example

        while True:
            sample_i = sample_rng.choice(self.dataset_size)

            curr_rng = np.random.RandomState(sample_i)

            target = curr_rng.uniform() < self.p_balance

            # Sample targets
            if self.num_relevant_contexts is None:
                context_labels = (
                    curr_rng.uniform(size=(self.num_contexts,)) < self.p_balance
                ).astype(int)
            else:
                context_labels = [target] * self.num_relevant_contexts + [
                    1 - target
                ] * (self.num_contexts - self.num_relevant_contexts)
                context_labels = curr_rng.permutation(context_labels)

            # Sample prototype indices
            if self.conditioning == "none":
                prototype_idxes = curr_rng.choice(
                    self.num_classes, p=weights, size=(self.num_contexts + 1,)
                )
            elif self.conditioning in ["iw", "in_weight"]:
                query_idx = curr_rng.choice(
                    self.num_high_freq_classes, p=weights_high_freq
                )
                example_idxes = (
                    curr_rng.choice(
                        self.num_low_freq_classes,
                        p=weights_low_freq,
                        size=(self.num_contexts,),
                    )
                    + self.num_high_freq_classes
                )
                prototype_idxes = np.concatenate((example_idxes, [query_idx]))
            elif self.conditioning in ["ic", "in_context"]:
                query_idx = (
                    curr_rng.choice(self.num_low_freq_classes, p=weights_low_freq)
                    + self.num_high_freq_classes
                )
                example_idxes = curr_rng.choice(
                    self.num_classes, p=weights, size=(self.num_contexts,)
                )

                # Find context label that is equal to target and set its prototype to be same as the query prototype
                if target not in context_labels:
                    context_labels[curr_rng.choice(self.num_contexts)] = target
                match_idx = curr_rng.choice(np.where(context_labels == target)[0])
                prototype_idxes = np.concatenate((example_idxes, [query_idx]))
            else:
                raise NotImplementedError

            label = np.concatenate((context_labels, [target]))

            # Fetch prototypes
            example = np.stack(
                [create_example(prototype_idx) for prototype_idx in prototype_idxes]
            ) * ((-1) ** label[:, None])
            example = (
                example + curr_rng.standard_normal(example.shape) * self.input_noise_std
            )

            if self.conditioning == "in_context":
                example[match_idx] = example[-1]

            # Label noise
            flip_mask = (curr_rng.uniform(size=label.shape) < self.label_noise).astype(
                int
            )
            label = flip_mask * (1 - label) + (1 - flip_mask) * label
            one_hot = np.zeros((self.num_contexts + 1, 2))
            one_hot[np.arange(self.num_contexts + 1), label] = 1

            if self.flip_label:
                one_hot = 1 - one_hot

            yield {
                "example": example,
                "label": one_hot,
            }
