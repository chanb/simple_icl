import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np


class StreamBlockBiUniform:
    def __init__(
        self,
        num_high_prob_classes: int,
        num_low_prob_classes: int,
        high_prob: float,
        num_dims: int,
        seed: int,
        linearly_separable: bool = False,
        margin: float = 0.2,
    ):
        assert 0.0 < high_prob < 1.0
        assert (
            high_prob / num_high_prob_classes >= (1 - high_prob) / num_low_prob_classes
        )
        self.num_high_prob_classes = num_high_prob_classes
        self.num_low_prob_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.high_prob = high_prob
        self.low_prob = 1 - high_prob
        self.num_dims = num_dims
        self.rng = np.random.RandomState(seed)

        if linearly_separable:
            pos = self.rng.standard_normal(
                size=(self.num_dims, 1),
            )
            pos /= np.linalg.norm(pos)
            neg = -pos

            done_generation = False
            high_prob_centers = np.zeros((self.num_high_prob_classes, self.num_dims))
            replace_mask = high_prob_centers == 0

            while not done_generation:
                new_samples = (
                    self.rng.standard_normal(
                        size=(self.num_high_prob_classes, self.num_dims)
                    )
                    * (1 - margin**2)
                    + pos.T
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                high_prob_centers = (
                    high_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = high_prob_centers @ pos
                replace_mask = dists < margin
                done_generation = np.sum(replace_mask) == 0

            done_generation = False
            low_prob_centers = np.zeros((self.num_low_prob_classes, self.num_dims))
            replace_mask = low_prob_centers == 0
            while not done_generation:
                new_samples = (
                    self.rng.standard_normal(
                        size=(self.num_low_prob_classes, self.num_dims)
                    )
                    * (1 - margin**2)
                    + neg.T
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                low_prob_centers = (
                    low_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = low_prob_centers @ neg
                replace_mask = dists < margin
                done_generation = np.sum(replace_mask) == 0

            self.centers = np.concatenate((high_prob_centers, low_prob_centers), axis=0)
        else:
            self.centers = self.rng.standard_normal(
                size=(self.num_classes, self.num_dims)
            )
            self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Discrete(2)

    def get_iid_context_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        abstract_class: int = 0,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        # Only do IID context
        while True:
            labels = self.rng.choice(
                self.num_classes,
                size=(num_examples + 1,),
                p=weights,
            )

            inputs = self.centers[labels]
            inputs += input_noise_std * self.rng.randn(*inputs.shape)

            if abstract_class:
                # Class 0 if high-prob lusters, class 1 otherwise
                # TODO: Maybe there can be an ablation on varying number of classes?
                labels = [int(label < self.num_high_prob_classes) for label in labels]
                labels = np.eye(2)[labels]
            else:
                labels = np.eye(self.num_classes)[labels]

            yield {
                "example": inputs,
                "label": labels,
                "flip_label": 0,
            }

    def get_non_iid_stratified_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        abstract_class: int = 0,
        stratified: int = 0,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        start_pos = fixed_start_pos
        low_prob_classes_sample_counts = np.zeros(self.num_low_prob_classes)
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            block_labels = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

            # Stratified sampling
            # Choose low prob. class as query and removes it from being sampled onwards
            available_low_prob_classes = np.where(
                low_prob_classes_sample_counts < stratified
            )[0]
            if len(available_low_prob_classes) and self.rng.rand() >= self.high_prob:
                query_label = self.rng.choice(available_low_prob_classes)
                low_prob_classes_sample_counts[query_label] += 1
                query_label += self.num_high_prob_classes
            else:
                query_label = self.rng.choice(
                    self.num_high_prob_classes,
                    size=(1,),
                )
            block_labels[-1] = query_label

            labels = [block_labels[0]] * (num_examples - start_pos) + [
                block_labels[1]
            ] * (start_pos + 1)

            inputs = self.centers[labels]
            inputs += input_noise_std * self.rng.randn(*inputs.shape)

            if abstract_class:
                # Class 0 if high-prob lusters, class 1 otherwise
                # TODO: Maybe there can be an ablation on varying number of classes?
                labels = [int(label < self.num_high_prob_classes) for label in labels]
                labels = np.eye(2)[labels]
            else:
                labels = np.eye(self.num_classes)[labels]

            yield {
                "example": inputs,
                "label": labels,
                "flip_label": 0,
            }

    def get_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        abstract_class: int = 0,
        sample_low_prob_class_only: int = 0,
        sample_high_prob_class_only: int = 0,
        flip_label: int = 0,
        scramble_context: int = 0,
        sample_relevant_context: int = 0,
        sample_irrelevant_context: int = 0,
    ):
        assert sample_low_prob_class_only + sample_high_prob_class_only <= 1

        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        start_pos = fixed_start_pos
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            if sample_irrelevant_context:
                start_pos = 0

            block_freq = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

            # TODO: Condition based on relevant vs non-relevant
            if sample_low_prob_class_only:
                # Sample low prob. class as query only
                block_freq[-1] = (
                    self.rng.choice(
                        self.num_low_prob_classes,
                        size=(1,),
                    )
                    + self.num_high_prob_classes
                )
            elif sample_high_prob_class_only:
                # Sample low prob. class as query only
                block_freq[-1] = self.rng.choice(
                    self.num_high_prob_classes,
                    size=(1,),
                )

            if sample_relevant_context and start_pos == 0:
                block_freq[0] = block_freq[1]
            elif sample_irrelevant_context and start_pos == 0:
                if abstract_class:
                    block_1_is_low_prob = block_freq[1] >= self.num_high_prob_classes
                    block_freq[0] = [
                        self.rng.choice(
                            self.num_high_prob_classes,
                            size=(1,),
                        ),
                        self.rng.choice(
                            self.num_low_prob_classes,
                            size=(1,),
                        )
                        + self.num_high_prob_classes,
                    ][1 - block_1_is_low_prob]

                else:
                    while block_freq[0] == block_freq[1]:
                        block_freq[0] = self.rng.choice(self.num_classes, size=(1,))

            if abstract_class:
                # Class 0 if high-prob clusters, class 1 otherwise
                block_labels = [
                    self.rng.choice(
                        self.num_high_prob_classes,
                        size=(num_examples + 1,),
                    ),
                    self.rng.choice(
                        self.num_low_prob_classes,
                        size=(num_examples + 1,),
                    )
                    + self.num_high_prob_classes,
                ]

                labels = np.concatenate(
                    (
                        block_labels[int(block_freq[0] >= self.num_high_prob_classes)][
                            : num_examples - start_pos
                        ],
                        block_labels[int(block_freq[1] >= self.num_high_prob_classes)][
                            num_examples - start_pos :
                        ],
                    )
                )
                inputs = self.centers[labels]
                inputs += input_noise_std * self.rng.randn(*inputs.shape)
                if flip_label:
                    labels = [
                        1 - int(label >= self.num_high_prob_classes) for label in labels
                    ]
                else:
                    labels = [
                        int(label >= self.num_high_prob_classes) for label in labels
                    ]
                labels = np.eye(2)[labels]
            else:
                labels = [block_freq[0]] * (num_examples - start_pos) + [
                    block_freq[1]
                ] * (start_pos + 1)
                inputs = self.centers[labels]
                inputs += input_noise_std * self.rng.randn(*inputs.shape)
                labels = np.eye(self.num_classes)[labels]

            if scramble_context:
                inds = self.rng.permutation(num_examples)
                inputs[:-1] = inputs[:-1][inds]
                labels[:-1] = labels[:-1][inds]

            yield {
                "example": inputs,
                "label": labels,
                "flip_label": int(flip_label),
            }
