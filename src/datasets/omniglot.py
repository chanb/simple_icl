import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gymnasium import spaces

import jax.random as jrandom
import numpy as np
import timeit
import torchvision.datasets as torch_datasets


IMAGE_SIZE = 105
N_CHARACTER_CLASSES = 1623
N_EXEMPLARS_PER_CLASS = 20
N_TRAIN_CLASSES = 964
N_TEST_CLASSES = 659


class Omniglot:
    def __init__(
        self,
        dataset_size: int,
        num_contexts: int,
        num_high_prob_classes: int,
        num_low_prob_classes: int,
        p_high: float,
        p_relevant_context: float,
        seed: int,
        train: bool,
        conditioning: str = "none",
        input_noise_std: float = 0.0,
        label_noise: float = 0.0,
        num_relevant_contexts: int = None,
        exemplar: str = "single",
        flip_label: bool = False,
    ):
        assert 0.0 < p_high < 1.0
        assert p_high / num_high_prob_classes >= (1 - p_high) / num_low_prob_classes
        assert num_high_prob_classes + num_low_prob_classes <= dataset_size
        assert num_relevant_contexts is None or num_relevant_contexts > 0

        self.num_high_prob_classes = num_high_prob_classes
        self.num_low_prob_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.p_high = p_high
        self.p_relevant_context = p_relevant_context
        self.low_prob = 1 - p_high
        self.dataset_size = dataset_size
        self.train = train
        self.seed = seed if train else seed + 1
        self.input_noise_std = input_noise_std
        self.label_noise = label_noise
        self.num_contexts = num_contexts
        self.num_relevant_contexts = num_relevant_contexts
        self.conditioning = conditioning
        self.rng = np.random.RandomState(seed)
        self.exemplar = exemplar
        self.flip_label = flip_label

        data_dir = os.path.join(os.environ["HOME"], "torch_datasets")
        download = True
        if "SLURM_TMPDIR" in os.environ:
            data_dir = os.path.join(os.environ["SLURM_TMPDIR"], "torch_datasets")
            download = False

        self.train_dataset = torch_datasets.Omniglot(
            data_dir,
            background=True,
            download=download,
        )
        self.test_dataset = torch_datasets.Omniglot(
            data_dir,
            background=False,
            download=download,
        )

        self._generate_dataset()

    @property
    def input_space(self):
        return spaces.Box(low=0, high=255, shape=(105, 105, 1), dtype=int)

    @property
    def output_space(self):
        return spaces.Discrete(self.num_classes)

    def _generate_dataset(self):
        print("generating dataset")
        tic = timeit.default_timer()

        weights = [
            self.p_high / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        rng = np.random.RandomState(self.seed)

        self.targets = rng.choice(
            self.num_classes,
            size=(self.dataset_size, self.num_contexts + 1),
            p=weights,
        )

        self.targets[rng.permutation(self.dataset_size)[: self.num_classes], -1] = (
            np.arange(self.num_classes)
        )
        self.swap_labels = rng.uniform(size=self.targets.shape) < self.label_noise

        if self.conditioning == "high_prob":
            self.targets[..., -1] = rng.choice(
                self.num_high_prob_classes, size=(self.dataset_size,)
            )
        elif self.conditioning == "low_prob":
            self.targets[..., -1] = (
                rng.choice(self.num_low_prob_classes, size=(self.dataset_size,))
                + self.num_high_prob_classes
            )

        relevant_context_mask = rng.choice(
            2,
            size=(self.dataset_size,),
            p=[1 - self.p_relevant_context, self.p_relevant_context],
        )

        context_from_query = np.sum(
            self.targets[:, :-1] == self.targets[:, [-1]], axis=-1
        )

        relevant_context_idxes = np.where(relevant_context_mask == 1)[0]
        if self.num_relevant_contexts is None:
            no_context_from_query_idxes = np.where(
                context_from_query[relevant_context_idxes] == 0
            )[0]

            self.targets[relevant_context_idxes[no_context_from_query_idxes], -1] = (
                rng.choice(self.num_contexts, size=(len(no_context_from_query_idxes),))
            )

        else:
            self.targets[relevant_context_idxes, : self.num_relevant_contexts] = (
                self.targets[relevant_context_idxes, -1][..., None]
            )

            if self.num_relevant_contexts != self.num_contexts:
                no_context_from_query_idxes = np.where(
                    context_from_query[relevant_context_idxes]
                    != self.num_relevant_contexts
                )[0]
                while len(no_context_from_query_idxes) > 0:
                    self.targets[
                        relevant_context_idxes[no_context_from_query_idxes],
                        self.num_relevant_contexts : -1,
                    ] = rng.choice(
                        self.num_classes,
                        size=(
                            len(no_context_from_query_idxes),
                            self.num_contexts - self.num_relevant_contexts,
                        ),
                        p=weights,
                    )
                    context_from_query = np.sum(
                        self.targets[:, self.num_relevant_contexts : -1]
                        == self.targets[:, [-1]],
                        axis=-1,
                    )

                    no_context_from_query_idxes = np.where(
                        context_from_query[relevant_context_idxes] > 0
                    )[0]
            self.targets = np.random.default_rng(self.seed).permuted(
                self.targets, axis=-1
            )

        irrelevant_context_idxes = np.where(relevant_context_mask == 0)[0]
        has_context_from_query_idxes = np.where(
            context_from_query[irrelevant_context_idxes] > 0
        )[0]
        while len(has_context_from_query_idxes) > 0:
            self.targets[
                irrelevant_context_idxes[has_context_from_query_idxes], :-1
            ] = rng.choice(
                self.num_classes,
                size=(len(has_context_from_query_idxes), self.num_contexts),
                p=weights,
            )
            context_from_query = np.sum(
                self.targets[:, :-1] == self.targets[:, [-1]], axis=-1
            )
            has_context_from_query_idxes = np.where(
                context_from_query[irrelevant_context_idxes] > 0
            )[0]

        toc = timeit.default_timer()
        print("dataset generation took {}s".format(toc - tic))

    def get_image(self, target, sample_i, context_i):
        rng = np.random.RandomState(
            self.seed * self.num_contexts * sample_i + context_i
        )

        offset = 0
        if self.exemplar != "single":
            offset = rng.randint(0, N_CHARACTER_CLASSES)

        idx = target * N_EXEMPLARS_PER_CLASS + offset
        image = (
            self.train_dataset[idx]
            if target < N_TRAIN_CLASSES
            else self.test_dataset[idx - N_EXEMPLARS_PER_CLASS * N_TRAIN_CLASSES]
        )[0]
        image = np.array(image)[..., None].astype(np.float32) / 255.0
        image += self.input_noise_std * rng.randn(*image.shape)
        return image

    def get_sequences(
        self,
        flip_label: int = 0,
    ):
        while True:
            sample_i = self.rng.choice(self.dataset_size)

            label = self.targets[sample_i]

            # Get example and reshape to (N, H, W, C) and normalize
            example = np.stack(
                [
                    self.get_image(target_i, sample_i, context_i)
                    for context_i, target_i in enumerate(label)
                ]
            )

            # OOD labels: Make sure OOD label is still within the same frequency class
            if flip_label:
                high_prob_class_idxes = np.where(label < self.num_high_prob_classes)[0]
                low_prob_class_idxes = np.where(label >= self.num_high_prob_classes)[0]
                label[high_prob_class_idxes] = (
                    label[high_prob_class_idxes] + 1
                ) % self.num_high_prob_classes
                label[low_prob_class_idxes] = (
                    label[low_prob_class_idxes] - self.num_high_prob_classes + 1
                ) % self.num_low_prob_classes + self.num_high_prob_classes

            # Get label distribution
            target = label[-1]
            label_dist = np.zeros(self.num_classes)
            label_dist[target] = 1 - self.label_noise
            label_dist[(target + 1) % self.num_classes] = self.label_noise

            # Label noise
            swap_label = self.swap_labels[sample_i]
            label = np.where(swap_label, (label + 1) % self.num_classes, label)

            one_hot = np.zeros((self.num_contexts + 1, self.num_classes))
            one_hot[np.arange(self.num_contexts + 1), label] = 1
            yield {
                "example": example,
                "label": one_hot,
                "label_dist": label_dist,
            }

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, sample_i):
        label = self.targets[sample_i]

        # Get example and reshape to (N, H, W, C) and normalize
        example = np.zeros((self.num_contexts + 1, *self.input_space.shape))
        for context_i, target_i in enumerate(label):
            example[context_i] = self.get_image(target_i, sample_i, context_i)

        # OOD labels: Make sure OOD label is still within the same frequency class
        if self.flip_label:
            high_prob_class_idxes = np.where(label < self.num_high_prob_classes)[0]
            low_prob_class_idxes = np.where(label >= self.num_high_prob_classes)[0]
            label[high_prob_class_idxes] = (
                label[high_prob_class_idxes] + 1
            ) % self.num_high_prob_classes
            label[low_prob_class_idxes] = (
                label[low_prob_class_idxes] - self.num_high_prob_classes + 1
            ) % self.num_low_prob_classes + self.num_high_prob_classes

        # Get label distribution
        target = label[-1]
        label_dist = np.zeros(self.num_classes)
        label_dist[target] = 1 - self.label_noise
        label_dist[(target + 1) % self.num_classes] = self.label_noise

        # Label noise
        swap_label = self.swap_labels[sample_i]
        label = np.where(swap_label, (label + 1) % self.num_classes, label)

        one_hot = np.zeros((self.num_contexts + 1, self.num_classes))
        one_hot[np.arange(self.num_contexts + 1), label] = 1

        return {
            "example": example,
            "target": one_hot,
            "label_dist": label_dist,
        }
