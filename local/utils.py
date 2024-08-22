from jaxl.constants import *
from jaxl.datasets import get_dataset

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import os
import timeit

from sklearn.metrics import confusion_matrix


# Get model predictions
def get_preds_labels(model, params, prefetched_data, max_label=None):
    all_preds = []
    all_labels = []
    all_outputs = []
    num_query_class_in_context = []

    for batch_i, data in enumerate(prefetched_data["samples"]):
        context_inputs = data["context_inputs"]
        context_outputs = data["context_outputs"]
        queries = data["queries"]
        one_hot_labels = data["outputs"]

        if hasattr(context_inputs, "numpy"):
            context_inputs = context_inputs.numpy()
            context_outputs = context_outputs.numpy()
            queries = queries.numpy()
            one_hot_labels = one_hot_labels.numpy()

        outputs, _, _ = model.forward(
            params[CONST_MODEL_DICT][CONST_MODEL],
            queries,
            {
                CONST_CONTEXT_INPUT: context_inputs,
                CONST_CONTEXT_OUTPUT: context_outputs,
            },
            eval=True,
        )

        if max_label is None:
            preds = np.argmax(outputs, axis=-1)
        else:
            preds = np.argmax(outputs[..., :max_label], axis=-1)
        labels = np.argmax(one_hot_labels, axis=-1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_outputs.append(outputs)
        num_query_class_in_context.append(
            np.max(np.argmax(context_outputs, axis=-1) == labels[:, None], axis=-1)
        )

    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    num_query_class_in_context = np.concatenate(num_query_class_in_context)
    return all_preds, all_labels, all_outputs, num_query_class_in_context


# Check model accuracy
def print_performance_with_aux(
    all_outputs,
    all_preds,
    all_labels,
    num_query_class_in_context,
    output_dim,
    context_len,
    fixed_length=True,
):
    conf_mat = confusion_matrix(all_labels, all_preds, labels=np.arange(output_dim))
    auxes = {}
    acc = np.trace(conf_mat) / np.sum(conf_mat) * 100
    loss = np.mean(
        optax.softmax_cross_entropy(
            all_outputs, jax.nn.one_hot(all_labels, num_classes=output_dim)
        )
    )
    auxes["all"] = {
        "accuracy": acc,
        "loss": loss,
        "query_class_in_context_ratio": np.mean(num_query_class_in_context),
    }

    return auxes


# Get dataloader
def get_data_loader(
    config,
    seed,
    visualize=False,
):
    dataset = get_dataset(
        config.learner_config.dataset_config,
        seed,
    )

    if visualize:
        plot_examples(dataset)

    data_loader = dataset.get_dataloader(config.learner_config)
    return dataset, data_loader


# Complete evaluation
def evaluate(
    model,
    params,
    prefetched_data,
    max_label,
    context_len,
    fixed_length=True,
):
    preds, labels, outputs, num_query_class_in_context = get_preds_labels(
        model, params, prefetched_data, max_label
    )
    auxes = print_performance_with_aux(
        outputs,
        preds,
        labels,
        num_query_class_in_context,
        prefetched_data["dataset_output_dim"],
        context_len,
        fixed_length,
    )
    return auxes["all"]["accuracy"], auxes["all"]["loss"], auxes
