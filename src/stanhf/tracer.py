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
        lines = [f"{r} // {log}" for r in res.split("\n")]
        return "\n".join(lines)

    return wrapped


def mergetraced(list_):
    """
    @returns Merged list of dictionaries & deepmerged metadata entries
    """
    list_ = [item for item in list_ if item is not None]
    merged = {k: v for d in list_ for k, v in d.items()}
    metadata = [d.get(METADATA, {}) for d in list_]
    merged[METADATA] = {k: v for d in metadata for k, v in d.items()}
    return merged
