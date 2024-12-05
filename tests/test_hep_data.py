"""
Test conversion of big model
============================
"""

import os

import pyhf.contrib.utils

from stanhf import convert, validate, build


pyhf.contrib.utils.download("https://doi.org/10.17182/hepdata.90607.v3/r3", "1Lbb-likelihoods")


def test_convert():
    """
    Convert massive hep-data model
    """
    root = convert("1Lbb-likelihoods/BkgOnly.json")
