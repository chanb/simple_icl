import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from types import SimpleNamespace
from typing import Any, Dict, Iterator, Union

import numpy as np
import random

from src.constants import *


def get_device(device: str):
    (device_name, *device_ids) = device.split(":")
    if device_name == CONST_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_name == CONST_GPU:
        assert (
            len(device_ids) > 0
        ), f"at least one device_id is needed, got {device_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
    else:
        raise ValueError(f"{device_name} is not a supported device.")


def set_seed(seed: int = 0):
    """
    Sets the random number generators' seed.

    :param seed: the seed
    :type seed: int:  (Default value = 0)

    """
    random.seed(seed)
    np.random.seed(seed)


def parse_dict(d: Dict) -> SimpleNamespace:
    """
    Parse dictionary into a namespace.
    Reference: https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace

    :param d: the dictionary
    :type d: Dict
    :return: the namespace version of the dictionary's content
    :rtype: SimpleNamespace

    """
    x = SimpleNamespace()
    _ = [
        setattr(x, k, parse_dict(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


def flatten_dict(d: Union[Dict, Any], label: str = None) -> Iterator:
    """
    Flattens a dictionary.

    :param d: the dictionary
    :param label: the parent's key name
    :type d: Dict
    :type label: str:  (Default value = None)
    :return: an iterator that yields the key-value pairs
    :rtype: Iterator

    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten_dict(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, d)


class DummySummaryWriter:
    """
    A fake SummaryWriter class for Tensorboard.
    """

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass
