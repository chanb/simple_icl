from gymnasium import spaces
from types import SimpleNamespace
from typing import Any

import chex
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDataset:
    def __init__(self, dataset, input_space, output_space):
        self._dataset = dataset
        self._input_space = input_space
        self._output_space = output_space

    @property
    def dataset(self):
        return self._dataset

    @property
    def input_space(self):
        return self._input_space

    @property
    def output_space(self):
        return self._output_space


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
            boundary = self.rng.uniform(
                low=-1.0,
                high=1.0,
                size=(self.num_dims, 1),
            )
            boundary /= np.linalg.norm(boundary)

            done_generation = False
            high_prob_centers = np.zeros((self.num_high_prob_classes, self.num_dims))
            replace_mask = high_prob_centers == 0

            while not done_generation:
                new_samples = self.rng.standard_normal(
                    size=(self.num_high_prob_classes, self.num_dims)
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                high_prob_centers = (
                    high_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = high_prob_centers @ boundary
                replace_mask = dists > -margin
                done_generation = np.sum(replace_mask) == 0
            print("Generated high prob centers")

            done_generation = False
            low_prob_centers = np.zeros((self.num_low_prob_classes, self.num_dims))
            replace_mask = low_prob_centers == 0
            while not done_generation:
                new_samples = self.rng.standard_normal(
                    size=(self.num_low_prob_classes, self.num_dims)
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                low_prob_centers = (
                    low_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = low_prob_centers @ boundary
                replace_mask = dists < margin
                done_generation = np.sum(replace_mask) == 0
            print("Generated low prob centers")

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

            block_freq = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

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
                        1 - int(label < self.num_high_prob_classes) for label in labels
                    ]
                else:
                    labels = [
                        int(label < self.num_high_prob_classes) for label in labels
                    ]
                labels = np.eye(2)[labels]
            else:
                labels = [block_freq[0]] * (num_examples - start_pos) + [
                    block_freq[1]
                ] * (start_pos + 1)
                inputs = self.centers[labels]
                inputs += input_noise_std * self.rng.randn(*inputs.shape)
                labels = np.eye(self.num_classes)[labels]

            yield {
                "example": inputs,
                "label": labels,
                "flip_label": int(flip_label),
            }


def get_dataset(
    # For get_sequences
    num_examples: int,
    input_noise_std: float,
    fixed_start_pos: int = -1,
    abstract_class: int = 0,
    sample_low_prob_class_only: int = 0,
    sample_high_prob_class_only: int = 0,
    stratified: int = 0,
    flip_label: int = 0,
    # For constructor
    num_high_prob_classes: int = 16,
    num_low_prob_classes: int = 256,
    high_prob: float = 0.8,
    num_dims: int = 64,
    mode: str = "default",
    seed: int = 42,
    linearly_separable: bool = False,
    margin: float = 0.2,
):
    if abstract_class:
        num_classes = 2
    else:
        num_classes = num_low_prob_classes + num_high_prob_classes
    task = StreamBlockBiUniform(
        num_high_prob_classes,
        num_low_prob_classes,
        high_prob,
        num_dims,
        seed,
        linearly_separable,
        margin,
    )

    if mode == "iid_context":
        seq_generator = task.get_iid_context_sequences
        args = (
            num_examples,
            input_noise_std,
            abstract_class,
        )
    elif mode == "non_iid_stratified":
        seq_generator = task.get_non_iid_stratified_sequences
        args = (
            num_examples,
            input_noise_std,
            fixed_start_pos,
            abstract_class,
            stratified,
        )
    elif mode == "default":
        seq_generator = task.get_sequences
        args = (
            num_examples,
            input_noise_std,
            fixed_start_pos,
            abstract_class,
            sample_low_prob_class_only,
            sample_high_prob_class_only,
            flip_label,
        )
    else:
        raise NotImplementedError

    dataset = tf.data.Dataset.from_generator(
        seq_generator,
        args=args,
        output_signature={
            "example": tf.TensorSpec(
                shape=(num_examples + 1, num_dims), dtype=tf.dtypes.float32
            ),
            "label": tf.TensorSpec(
                shape=(num_examples + 1, num_classes), dtype=tf.dtypes.int32
            ),
            "flip_label": tf.TensorSpec(shape=[], dtype=tf.dtypes.int8),
        },
    )
    return TFDataset(
        dataset,
        task.input_space,
        task.output_space,
    )


def prepare_seqs_for_icl(ds, num_classes: int):
    """Convert example and label sequences for use by the transformer."""

    def _convert_dict(example):
        # Cast the examples into the correct shape and tf datatype.
        examples = tf.cast(example["example"], tf.float32)

        # Cast the labels into the correct tf datatype.
        targets = tf.cast(example["label"], tf.int32)  # (B,SS)

        flip_labels = tf.cast(example["flip_label"], tf.int8)

        # Just use the original sequence of labels, e.g. [label, label, ...]
        is_one_hot = targets.shape[-1] == num_classes
        if not is_one_hot:
            targets = tf.one_hot(targets, num_classes)  # (B,SS)

        # ret_dict = {"examples": examples, "labels": labels, "target": targets}
        ret_dict = {
            "example": examples,
            "target": targets,
            "flip_label": flip_labels,
        }
        return tf.data.Dataset.from_tensors(ret_dict)

    return ds.flat_map(_convert_dict)


def get_data_loader(config: SimpleNamespace) -> Any:
    dataset_kwargs = config.dataset_kwargs
    dataset = get_dataset(
        num_examples=dataset_kwargs.num_examples,
        input_noise_std=dataset_kwargs.input_noise_std,
        num_high_prob_classes=dataset_kwargs.num_high_prob_classes,
        num_low_prob_classes=dataset_kwargs.num_low_prob_classes,
        high_prob=dataset_kwargs.high_prob,
        num_dims=dataset_kwargs.num_dims,
        mode=dataset_kwargs.mode,
        seed=config.data_seed,
        linearly_separable=getattr(dataset_kwargs, "linearly_separable", False),
        margin=getattr(dataset_kwargs, "margin", 0.2),
        fixed_start_pos=getattr(dataset_kwargs, "fixed_start_pos", -1),
        abstract_class=getattr(dataset_kwargs, "abstract_class", 0),
        sample_low_prob_class_only=getattr(
            dataset_kwargs, "sample_low_prob_class_only", 0
        ),
        sample_high_prob_class_only=getattr(
            dataset_kwargs, "sample_high_prob_class_only", 0
        ),
        stratified=getattr(dataset_kwargs, "stratified", 0),
        flip_label=getattr(dataset_kwargs, "flip_label", False),
    )
    ds_seqs = dataset.dataset

    shuffle_buffer_size = config.shuffle_buffer_size
    ds = ds_seqs.batch(config.batch_size).prefetch(config.num_workers)
    ds = prepare_seqs_for_icl(
        ds,
        dataset.output_space.n,
    )
    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size)
    return tfds.as_numpy(ds), dataset
