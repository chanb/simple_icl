import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
srcdir = os.path.join(os.path.dirname(parentdir), "src")
sys.path.insert(0, parentdir)
sys.path.insert(0, srcdir)

import src.models as models

from src.constants import *

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

    for batch_i, batch in enumerate(prefetched_data["samples"]):
        outputs, updates = model.forward(
            params[CONST_MODEL],
            batch,
            eval=True,
        )

        if max_label is None:
            preds = np.argmax(outputs, axis=-1)
        else:
            preds = np.argmax(outputs[..., :max_label], axis=-1)

        context_outputs = batch["target"][:, :-1]
        targets = batch["target"][:, -1]
        labels = np.argmax(targets, axis=-1)
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
