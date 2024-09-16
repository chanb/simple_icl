import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np
import timeit

"""
TODO:
- Add mode for training IW transformer: This should set context to zero.
"""

class Synthetic:
    def __init__(
        self,
        dataset_size: int,
        num_contexts: int,
        num_high_prob_classes: int,
        num_low_prob_classes: int,
        p_high: float,
        p_relevant_context: float,
        num_dims: int,
        seed: int,
        train: bool,
        conditioning: str = "none",
        input_noise_std: float = 0.0,
        label_noise: float = 0.0,
    ):
        assert 0.0 < p_high < 1.0
        assert (
            p_high / num_high_prob_classes >= (1 - p_high) / num_low_prob_classes
        )
        self.num_high_prob_classes = num_high_prob_classes
        self.num_low_prob_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.p_high = p_high
        self.p_relevant_context = p_relevant_context
        self.low_prob = 1 - p_high
        self.num_dims = num_dims
        self.dataset_size = dataset_size
        self.train = train
        self.seed = seed
        self.input_noise_std = input_noise_std
        self.label_noise = label_noise
        self.num_contexts = num_contexts
        self.conditioning = conditioning
        self.rng = np.random.RandomState(seed)

        self.centers = self.rng.standard_normal(
            size=(self.num_classes, self.num_dims)
        )
        self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)
        self._generate_dataset()

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Discrete(self.num_classes)

    def _generate_dataset(self):
        print("generating dataset")
        tic = timeit.default_timer()
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.p_high / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        if self.train:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random.RandomState(self.seed + 1)

        self.targets = rng.choice(
            self.num_classes,
            size=(self.dataset_size, self.num_contexts + 1),
            p=weights,
        )
        self.swap_labels = rng.uniform(size=self.targets.shape) < self.label_noise

        if self.conditioning == "high_prob":
            self.targets[..., -1] = rng.choice(
                self.num_high_prob_classes,
                size=(self.dataset_size,)
            )
        elif self.conditioning == "low_prob":
            self.targets[..., -1] = rng.choice(
                self.num_low_prob_classes,
                size=(self.dataset_size,)
            ) + self.num_high_prob_classes

        relevant_context_mask = rng.choice(
            2,
            size=(self.dataset_size,),
            p=[1 - self.p_relevant_context, self.p_relevant_context]
        )

        context_from_query = np.sum(self.targets[:, :-1] == self.targets[:, [-1]], axis=-1)

        relevant_context_idxes = np.where(relevant_context_mask == 1)[0]
        no_context_from_query_idxes = np.where(context_from_query[relevant_context_idxes] == 0)[0]

        while len(no_context_from_query_idxes) > 0:
            self.targets[relevant_context_idxes[no_context_from_query_idxes], :-1] = rng.choice(
                self.num_classes,
                size=(len(no_context_from_query_idxes), self.num_contexts),
                p=weights,
            )
            context_from_query = np.sum(self.targets[:, :-1] == self.targets[:, [-1]], axis=-1)
            no_context_from_query_idxes = np.where(context_from_query[relevant_context_idxes] == 0)[0]

        irrelevant_context_idxes = np.where(relevant_context_mask == 0)[0]
        has_context_from_query_idxes = np.where(context_from_query[irrelevant_context_idxes] > 0)[0]
        while len(has_context_from_query_idxes) > 0:
            self.targets[irrelevant_context_idxes[has_context_from_query_idxes], :-1] = rng.choice(
                self.num_classes,
                size=(len(has_context_from_query_idxes), self.num_contexts),
                p=weights,
            )
            context_from_query = np.sum(self.targets[:, :-1] == self.targets[:, [-1]], axis=-1)
            has_context_from_query_idxes = np.where(context_from_query[irrelevant_context_idxes] > 0)[0]

        self.inputs = self.centers[self.targets.flatten()].reshape(
            (self.dataset_size, self.num_contexts + 1, -1)
        )
        self.inputs += self.input_noise_std * rng.randn(*self.inputs.shape)
        toc = timeit.default_timer()
        print("dataset generation took {}s".format(toc - tic))

    def get_sequences(
        self,
        flip_label: int = 0,
        abstract_class: int = 0, # TODO: Fix this if we want binary
    ):
        num_classes = 2 if abstract_class else self.num_classes
        while True:
            sample_i = self.rng.choice(self.dataset_size)
            
            example = self.inputs[sample_i]

            swap_label = self.swap_labels[sample_i]
            label = self.targets[sample_i]
            label = np.where(swap_label, (label + 1) % self.num_classes, label)

            if flip_label:
                label = (label + 1) % self.num_classes
                
            label = np.eye(num_classes)[label]

            yield {
                "example": example,
                "label": label,
            }
