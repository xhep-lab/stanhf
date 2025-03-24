"""
Trace origin of lines in Stan code
==================================
"""

from os.path import basename
from functools import wraps


METADATA = "_metadata"


def metadata(func):
    """
    @returns Metadata about function
    """
    file_name = basename(func.__code__.co_filename)
    line = func.__code__.co_firstlineno
    return f"from {func.__qualname__} [{file_name}:L{line}]"


def add_metadata_entry(func):
    """
    @returns Result appended with metadata about origin
    """
    log = metadata(func)

    @wraps(func)
    def wrapped(other, *args, **kwargs):
        res = func(other, *args, **kwargs)
        res[METADATA] = res.get(METADATA, {})
        for k in res:
            if k != METADATA:
                res[METADATA][k] = log
        return res

    return wrapped


def add_metadata_comment(func):
    """
    @returns Result appended with metadata about origin
    """
    log = metadata(func)

    @wraps(func)
    def wrapped(other, *args, **kwargs):
        res = func(other, *args, **kwargs)
        lines = [f"{r} // {log}" for r in res.split("\n") if r.strip()]
        return "\n".join(lines)

    return wrapped


def shallow_merge(list_):
    """
    @returns Shallow-merge a list of dictionaries
    """
    return {k: v for d in list_ if d for k, v in d.items()}


def merge_metadata(list_):
    """
    @returns Shallow-merged list of dictionaries with shallow-merged metadata
    """
    merged = shallow_merge(list_)
    merged[METADATA] = shallow_merge([d.pop(METADATA, {}) for d in list_ if d])
    return merged
