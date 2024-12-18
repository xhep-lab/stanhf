"""
Trace origin of lines in Stan code
==================================
"""

from os.path import basename
from functools import wraps


METADATA = "_metadata"


def trace(func):
    """
    @returns Result appended with metadata about origin
    """
    file_name = basename(func.__code__.co_filename)
    line = func.__code__.co_firstlineno
    log = f"from {func.__qualname__} [{file_name}:L{line}]"

    @wraps(func)
    def wrapped(other, *args, **kwargs):
        res = func(other, *args, **kwargs)

        if isinstance(res, dict):

            res[METADATA] = res.get(METADATA, {})

            for k in res:
                if k != METADATA:
                    res[METADATA][k] = log

            return res

        if isinstance(res, str):

            lines = [f"{r} // {log}" for r in res.split("\n")]
            return "\n".join(lines)

        return res

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
