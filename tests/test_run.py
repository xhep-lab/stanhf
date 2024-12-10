"""
Test Stan target function
=========================
"""

import os

import numpy as np

from stanhf import convert, validate, build


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "example.json"))
RNG = np.random.default_rng(111)


def test_target():
    """
    Validate output from Stan against pyhf
    """
    root, par, fixed, null = convert(EXAMPLE)
    build(root)
    validate(root, RNG)
