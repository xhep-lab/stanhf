"""
Test Stan target function
=========================
"""

import os

import numpy as np

from stanhf import Convert


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "test.json"))
RNG = np.random.default_rng(111)


def test_target():
    """
    Validate output from Stan against pyhf
    """
    convert = Convert(EXAMPLE)
    convert.validate_par_names()
    convert.validate_target(rng=RNG)
