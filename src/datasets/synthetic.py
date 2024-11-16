import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import numpy as np
import timeit


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
        num_relevant_contexts: int = None,
    ):
        assert 0.0 < p_high < 1.0
        assert p_high / num_high_prob_classes >= (1 - p_high) / num_low_prob_classes
        assert num_high_prob_classes + num_low_prob_classes <= dataset_size
        assert num_relevant_contexts is None or num_relevant_contexts > 0

        self.num_high_freq_classes = num_high_prob_classes
        self.num_low_freq_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.p_high = p_high
        self.p_relevant_context = p_relevant_context
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
        self.rng = np.random.RandomState(seed)

        self.centers = self.rng.standard_normal(size=(self.num_classes, self.num_dims))
        self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)

    @property
    def input_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.num_dims,))

    @property
    def output_space(self):
        return spaces.Discrete(self.num_classes)

    def get_sequences(
        self,
        flip_label: int = 0,
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

        while True:
            sample_i = sample_rng.choice(self.dataset_size)

            curr_rng = np.random.RandomState(sample_i)

            if self.conditioning == "none":
                target = curr_rng.choice(self.num_classes, p=weights)
            elif self.conditioning == "high_prob":
                target = curr_rng.choice(self.num_high_freq_classes, p=weights_high_freq)
            elif self.conditioning == "low_prob":
                target = curr_rng.choice(self.num_low_freq_classes, p=weights_low_freq) + self.num_high_freq_classes

            context_labels = (
                curr_rng.choice(self.num_classes, p=weights, size=(self.num_contexts,))
            ).astype(int)

            # Decide context labels
            if self.num_relevant_contexts is not None:
                num_relevant = self.num_relevant_contexts
            else:
                num_relevant = np.sum(curr_rng.uniform(size=(self.num_contexts)) < self.p_relevant_context)

            num_relevant_in_context = len(np.where(context_labels == target)[0])

            if num_relevant > 0:
                num_replacement = num_relevant - num_relevant_in_context
                irrelevant_context_idx = curr_rng.permutation(np.where(context_labels != target)[0])[:num_replacement]
                context_labels[irrelevant_context_idx] = target
            else:
                relevant_context_idx = np.where(context_labels == target)[0]
                context_labels[relevant_context_idx] = (
                    curr_rng.choice(self.num_classes, p=weights, size=relevant_context_idx.shape)
                ).astype(int)

            # Get prototypes
            label = np.concatenate((context_labels, [target]))
            example = self.centers[label]
            example = (
                example + curr_rng.standard_normal(example.shape) * self.input_noise_std
            )

            # OOD labels: Make sure OOD label is still within the same frequency class
            if flip_label:
                high_freq_class_idxes = np.where(label < self.num_high_freq_classes)[0]
                low_freq_class_idxes = np.where(label >= self.num_high_freq_classes)[0]
                label[high_freq_class_idxes] = (
                    label[high_freq_class_idxes] + 1
                ) % self.num_high_freq_classes
                label[low_freq_class_idxes] = (
                    label[low_freq_class_idxes] - self.num_high_freq_classes + 1
                ) % self.num_low_freq_classes + self.num_high_freq_classes

            # Get label distribution
            label_dist = np.zeros(self.num_classes)
            label_dist[target] = 1 - self.label_noise
            label_dist[(target + 1) % self.num_classes] = self.label_noise

            # Label noise
            flip_mask = (curr_rng.uniform(size=label.shape) < self.label_noise).astype(
                int
            )
            label = np.where(flip_mask, (label + 1) % self.num_classes, label)

            one_hot = np.zeros((self.num_contexts + 1, self.num_classes))
            one_hot[np.arange(self.num_contexts + 1), label] = 1

            yield {
                "example": example,
                "label": one_hot,
                "label_dist": label_dist,
            }
