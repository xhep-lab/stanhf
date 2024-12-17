"""
Manipulate and parse strings and data
=====================================
"""

import json
import warnings

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
    if not args:
        return f"target += {dist}({var});"
    joined = ", ".join([str(a) for a in args])
    return f"target += {dist}({var} | {joined});"


def block(name, data):
    """
    @returns Stan program block
    """
    if data is None:
        return None
    if isinstance(data, list):
        data = "\n".join([d for d in data if d is not None])
    return f"{name}" + "{\n" + data + "\n}"


def flatten(list_):
    """
    @returns Flattened list
    """
    flat = []

    for item in list_:
        if isinstance(item, list):
            flat += item
        else:
            flat.append(item)

    return flat


def hashed(list_):
    """
    @returns Hash of nested lists
    """
    return hash(repr(list_))


def merge(list_):
    """
    @returns Merged list of dictionaries & deepmerged metadata entries
    """
    list_ = [item for item in list_ if item is not None]
    merged = {k: v for d in list_ for k, v in d.items()}
    metadata = [d.get(METADATA, {}) for d in list_]
    merged[METADATA] = {k: v for d in metadata for k, v in d.items()}
    return merged


def jlint(file_name):
    """
    @param file_name JSON file to be linted by indenting and sorting keys
    """
    with open(file_name, encoding="utf-8") as json_file:
        data = json.load(json_file)
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def read_par_bound(bound, size):
    """
    @returns Parameter bound as tuple
    """
    if size == 0:
        return (bound[0][0], bound[0][1])
    return ([b[0] for b in bound], [b[1] for b in bound])


def read_par_init(init, size):
    """
    @returns Parameter initial value as scalar if required
    """
    if size == 0:
        return init[0]
    return init


def read_observed(observed):
    """
    @returns Parameter initial value as scalar if required
    """
    int_observed = [int(o) for o in observed]

    if int_observed != observed:
        warnings.warn(
            f"observed converted to integer: {int_observed} vs. {observed}")

    return int_observed


def expand_par_name(name, size):
    """
    @returns Expanded names of parameter
    """
    if size == 0:
        return name
    return [f"{name}[{i}]" for i in range(size)]


def pyhf_par_names(pars):
    """
    @returns Make names in pyhf style
    """
    d = {}
    for n, s in pars.items():
        d.setdefault(n, 0)
        d[n] += s

    return flatten(expand_par_name(n, s) for n, s in d.items())


def pyhf_pars(pars):
    """
    @returns Make parameter values dictionary with keys in pyhf style
    """
    d = {}
    for k, v in pars.items():
        if isinstance(v, list):
            for n, e in zip(expand_par_name(k, len(v)), v):
                d[n] = e
        else:
            d[k] = v

    return d


def pyhf_order(order, pars):
    """
    @returns Expanded parameter names in pyhf order
    """
    stripped = [p.split("[", 1)[0] for p in pars if "[" in p]
    return flatten(expand_par_name(o, stripped.count(o)) for o in order)
