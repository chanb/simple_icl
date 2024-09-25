import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from prefetch_generator import BackgroundGenerator
from types import SimpleNamespace
from typing import Any

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

import src.datasets.omniglot as omniglot
import src.datasets.streamblock as streamblock
import src.datasets.synthetic as synthetic


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


def get_streamblock_seq_generator(
    dataset_kwargs: SimpleNamespace,
    seed: int,
):
    num_examples = dataset_kwargs.num_examples
    input_noise_std = dataset_kwargs.input_noise_std
    num_high_prob_classes = dataset_kwargs.num_high_prob_classes
    num_low_prob_classes = dataset_kwargs.num_low_prob_classes
    high_prob = dataset_kwargs.high_prob
    num_dims = dataset_kwargs.num_dims
    mode = dataset_kwargs.mode
    linearly_separable = getattr(dataset_kwargs, "linearly_separable", False)
    margin = getattr(dataset_kwargs, "margin", 0.2)
    fixed_start_pos = getattr(dataset_kwargs, "fixed_start_pos", -1)
    abstract_class = getattr(dataset_kwargs, "abstract_class", 0)
    sample_low_prob_class_only = getattr(
        dataset_kwargs, "sample_low_prob_class_only", 0
    )
    sample_high_prob_class_only = getattr(
        dataset_kwargs, "sample_high_prob_class_only", 0
    )
    stratified = getattr(dataset_kwargs, "stratified", 0)
    flip_label = getattr(dataset_kwargs, "flip_label", 0)
    scramble_context = getattr(dataset_kwargs, "scramble_context", 0)
    sample_relevant_context = getattr(dataset_kwargs, "sample_relevant_context", 0)
    sample_irrelevant_context = getattr(dataset_kwargs, "sample_irrelevant_context", 0)

    if abstract_class:
        num_classes = 2
    else:
        num_classes = num_low_prob_classes + num_high_prob_classes
    task = streamblock.StreamBlockBiUniform(
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
            scramble_context,
            sample_relevant_context,
            sample_irrelevant_context,
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


def get_omniglot_seq_generator(
    dataset_kwargs: SimpleNamespace,
    seed: int,
):
    task = omniglot.Omniglot(
        dataset_kwargs.dataset_size,
        dataset_kwargs.num_contexts,
        dataset_kwargs.num_high_prob_classes,
        dataset_kwargs.num_low_prob_classes,
        dataset_kwargs.p_high,
        dataset_kwargs.p_relevant_context,
        seed,
        dataset_kwargs.train,
        getattr(dataset_kwargs, "conditioning", "none"),
        getattr(dataset_kwargs, "input_noise_std", 0.0),
        getattr(dataset_kwargs, "label_noise", 0.0),
        getattr(dataset_kwargs, "num_relevant_contexts", None),
        getattr(dataset_kwargs, "exemplar", "single"),
    )

    num_classes = (
        dataset_kwargs.num_low_prob_classes + dataset_kwargs.num_high_prob_classes
    )

    dataset = tf.data.Dataset.from_generator(
        task.get_sequences,
        args=(getattr(dataset_kwargs, "flip_label", 0),),
        output_signature={
            "example": tf.TensorSpec(
                shape=(
                    dataset_kwargs.num_contexts + 1,
                    omniglot.IMAGE_SIZE,
                    omniglot.IMAGE_SIZE,
                    1,
                ),
                dtype=tf.dtypes.float32,
            ),
            "label": tf.TensorSpec(
                shape=(dataset_kwargs.num_contexts + 1, num_classes),
                dtype=tf.dtypes.int32,
            ),
            "label_dist": tf.TensorSpec(
                shape=(num_classes,),
                dtype=tf.dtypes.float32,
            ),
        },
    )
    return TFDataset(
        dataset,
        task.input_space,
        task.output_space,
    )


def get_synthetic_seq_generator(
    dataset_kwargs: SimpleNamespace,
    seed: int,
):
    task = synthetic.Synthetic(
        dataset_kwargs.dataset_size,
        dataset_kwargs.num_contexts,
        dataset_kwargs.num_high_prob_classes,
        dataset_kwargs.num_low_prob_classes,
        dataset_kwargs.p_high,
        dataset_kwargs.p_relevant_context,
        dataset_kwargs.num_dims,
        seed,
        dataset_kwargs.train,
        getattr(dataset_kwargs, "conditioning", "none"),
        getattr(dataset_kwargs, "input_noise_std", 0.0),
        getattr(dataset_kwargs, "label_noise", 0.0),
        getattr(dataset_kwargs, "num_relevant_contexts", None),
    )
    num_classes = (
        dataset_kwargs.num_low_prob_classes + dataset_kwargs.num_high_prob_classes
    )

    dataset = tf.data.Dataset.from_generator(
        task.get_sequences,
        args=(getattr(dataset_kwargs, "flip_label", 0),),
        output_signature={
            "example": tf.TensorSpec(
                shape=(dataset_kwargs.num_contexts + 1, dataset_kwargs.num_dims),
                dtype=tf.dtypes.float32,
            ),
            "label": tf.TensorSpec(
                shape=(dataset_kwargs.num_contexts + 1, num_classes),
                dtype=tf.dtypes.int32,
            ),
            "label_dist": tf.TensorSpec(
                shape=(num_classes,),
                dtype=tf.dtypes.float32,
            ),
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

        # Just use the original sequence of labels, e.g. [label, label, ...]
        is_one_hot = targets.shape[-1] == num_classes
        if not is_one_hot:
            targets = tf.one_hot(targets, num_classes)  # (B,SS)

        # ret_dict = {"examples": examples, "labels": labels, "target": targets}
        ret_dict = {
            "example": examples,
            "target": targets,
        }

        if "flip_label" in example:
            flip_labels = tf.cast(example["flip_label"], tf.int8)
            ret_dict["flip_label"] = flip_labels

        if "label_dist" in example:
            label_dists = tf.cast(example["label_dist"], tf.float32)
            ret_dict["label_dist"] = label_dists

        return tf.data.Dataset.from_tensors(ret_dict)

    return ds.flat_map(_convert_dict)


def get_iter(data_loader):
    loader = iter(data_loader)
    while True:
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(data_loader)
            batch = next(loader)

        for k, v in batch.items():
            if hasattr(v, "numpy"):
                batch[k] = v.numpy()
        yield jax.device_put(batch)


def get_data_loader(config: SimpleNamespace) -> Any:
    dataset_name = config.dataset_name
    dataset_kwargs = config.dataset_kwargs

    num_workers = getattr(config, "num_workers", 0)
    if dataset_name == "omniglot":
        from torch.utils.data import DataLoader

        batch_size = config.batch_size
        shuffle = True
        drop_last = True
        dataset = omniglot.Omniglot(
            dataset_kwargs.dataset_size,
            dataset_kwargs.num_contexts,
            dataset_kwargs.num_high_prob_classes,
            dataset_kwargs.num_low_prob_classes,
            dataset_kwargs.p_high,
            dataset_kwargs.p_relevant_context,
            config.seeds.data_seed,
            dataset_kwargs.train,
            getattr(dataset_kwargs, "conditioning", "none"),
            getattr(dataset_kwargs, "input_noise_std", 0.0),
            getattr(dataset_kwargs, "label_noise", 0.0),
            getattr(dataset_kwargs, "num_relevant_contexts", None),
            getattr(dataset_kwargs, "exemplar", "single"),
            getattr(dataset_kwargs, "flip_label", False),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    else:
        if dataset_name == "streamblock":
            seq_generator = get_streamblock_seq_generator
        elif dataset_name == "omniglot":
            seq_generator = get_omniglot_seq_generator
        elif dataset_name == "synthetic":
            seq_generator = get_synthetic_seq_generator

        dataset = seq_generator(dataset_kwargs, config.seeds.data_seed)
        ds_seqs = dataset.dataset

        shuffle_buffer_size = config.shuffle_buffer_size
        ds = ds_seqs.batch(config.batch_size).prefetch(num_workers)
        ds = prepare_seqs_for_icl(
            ds,
            dataset.output_space.n,
        )
        ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size)
        loader = tfds.as_numpy(ds)

    loader = get_iter(loader)
    loader = BackgroundGenerator(loader, max_prefetch=num_workers)

    return loader, dataset
