"""
Manipulate and parse strings and data
=====================================
"""

import json
import warnings


def join(*name):
    """
    @returns Snake-case convention for Stan variables
    """
    return "_".join(name)


def remove_prefix(name, prefix):
    """
    @returns Name with a prefix removed
    """
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def add_to_target(dist, var, *args):
    """
    @returns Stan contribution to target
    """
    joined = ", ".join([str(a) for a in args])
    return f"{var} ~ {dist}({joined});"


def block(name, data):
    """
    @returns Stan program block
    """
    if data is None:
        return None

    if isinstance(data, list):
        data = "\n".join([d for d in data if d is not None])

    if not data.strip():
        return None

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


def format_json_file(file_name):
    """
    @param file_name JSON file to be formatted by indenting and sorting keys
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
