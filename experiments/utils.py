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
import jax.random as jrandom
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
    all_auxes = dict()

    for batch_i, batch in enumerate(prefetched_data["samples"]):
        outputs, model_aux = model.forward(
            params[CONST_MODEL],
            batch,
            eval=True,
        )

        if max_label is None:
            preds = np.argmax(outputs, axis=-1)
        else:
            preds = np.argmax(outputs[..., :max_label], axis=-1)

        targets = batch["target"][:, -1]
        contexts = np.argmax(batch["target"][:, :-1], axis=-1)
        labels = np.argmax(targets, axis=-1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_outputs.append(outputs)

        for aux_key in model_aux:
            all_auxes.setdefault(aux_key, [])
            all_auxes[aux_key].append(model_aux[aux_key])

        if "label_dist" in batch:
            all_auxes.setdefault("label_dist", [])
            all_auxes["label_dist"].append(batch["label_dist"])

        all_auxes.setdefault("context contains query class", [])
        all_auxes["context contains query class"].append(
            np.sum(contexts == labels[:, None], axis=-1) > 0
        )

    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        ret_auxes = {k: np.concatenate(v) for k, v in all_auxes.items()}
    except:
        ret_auxes = {
            "context contains query class": np.concatenate(all_auxes["context contains query class"])
        }
        if "label_dist" in batch:
            ret_auxes["label_dist"] = np.concatenate(all_auxes["label_dist"])
    return all_preds, all_labels, all_outputs, ret_auxes


# Check model accuracy
def print_performance_with_aux(
    all_outputs,
    all_preds,
    all_labels,
    all_auxes,
    output_dim,
    sample_key,
    fixed_length=True,
):
    conf_mat = confusion_matrix(all_labels, all_preds, labels=np.arange(output_dim))
    auxes = {}
    acc = np.trace(conf_mat) / np.sum(conf_mat) * 100

    if "label_dist" in all_auxes:
        loss = np.mean(
            optax.softmax_cross_entropy(
                all_outputs, all_auxes["label_dist"]
            )
        )
    else:
        loss = np.mean(
            optax.softmax_cross_entropy(
                all_outputs, jax.nn.one_hot(all_labels, num_classes=output_dim)
            )
        )

    auxes = {
        "accuracy": acc,
        "loss": loss,
    }

    for aux_key in all_auxes:
        if aux_key == "alpha":
            continue
        elif aux_key == "h":
            auxes["similarity"] = np.mean(all_auxes[aux_key][..., 0], axis=0)
        elif aux_key == "p_iwl":
            auxes["p_iwl"] = np.mean(all_auxes[aux_key])
            auxes["num p_iwl >= 0.5"] = np.mean(all_auxes[aux_key] >= 0.5)
            auxes["p_iwl given context contains query class"] = np.mean(
                all_auxes[aux_key][np.where(all_auxes["context contains query class"])[0]] >= 0.5
            )
        elif aux_key in ["ic_pred", "iw_pred"]:
            preds = jax.vmap(
                lambda probs, key: jrandom.choice(key, output_dim, p=probs)
            )(
                all_auxes[aux_key],
                jrandom.split(sample_key, num=len(all_auxes[aux_key])),
            )
            auxes[aux_key] = np.mean(preds == all_labels)
        elif aux_key == "context contains query class":
            auxes["context contains query class"] = np.mean(all_auxes["context contains query class"])

    return auxes


# Complete evaluation
def evaluate(
    model,
    params,
    prefetched_data,
    max_label,
    sample_key,
    fixed_length=True,
):
    preds, labels, outputs, model_auxes = get_preds_labels(
        model, params, prefetched_data, max_label
    )
    auxes = print_performance_with_aux(
        outputs,
        preds,
        labels,
        model_auxes,
        prefetched_data["dataset_output_dim"],
        sample_key,
        fixed_length,
    )
    return auxes
