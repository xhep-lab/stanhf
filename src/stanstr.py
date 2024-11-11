"""
Manipulate and parse strings and data
=====================================
"""

import numpy as np
from .tracer import METADATA


def join(*name):
    """
    @returns Snake-case convention for Stan variables
    """
    return "_".join(name)


def add_to_target(dist, var, *args):
    """
    @returns Stan contribution to target
    """
    joined = ", ".join([str(a) for a in args])
    return f"target += {dist}({var} | {joined});"


def block(name, data):
    """
    @returns Stan program block
    """
    if isinstance(data, list):
        data = "\n".join([d for d in data if d is not None])
    return f"{name}" + "{\n" + data + "\n}"


def squeeze(list_):
    """
    @returns Squeeze a list
    """
    return np.squeeze(list_).tolist()


def flatten(list_):
    """
    @returns Flatten a list
    """
    return [e for r in list_ for e in np.atleast_1d(r) if r is not None]


def merge(list_):
    """
    @returns Merge a list of dictionaries & deepmerge metadata entries
    """
    list_ = [l for l in list_ if l is not None]
    merged = {k: v for d in list_ for k, v in d.items()}
    merged[METADATA] = {
        k: v for d in list_ for k,
        v in d.get(
            METADATA,
            {}).items()}
    return merged
