"""
Test conversion of big model
============================
"""

import pyhf.contrib.utils

from stanhf import convert


pyhf.contrib.utils.download("https://doi.org/10.17182/hepdata.90607.v3/r3", "1Lbb-likelihoods")


def test_convert():
    """
    Convert massive hep-data model
    """
    convert("1Lbb-likelihoods/BkgOnly.json")
