"""
Test conversion of big model
============================
"""

import os

import numpy as np

from stanhf import Convert
from test_hep_data import fetch_hep_data


RNG = np.random.default_rng(111)


def test_validate_hep_data_patch():
    """
    Test a HEP-DATA histfactory model
    """
    doi = "10.17182/hepdata.89408.v3/r2"
    folder_name = fetch_hep_data(doi)

    json_file = os.path.join(folder_name, "RegionC/BkgOnly.json")
    patch = os.path.join(folder_name, "RegionC/patchset.json")

    if not os.path.exists(patch):
        raise RuntimeError(f"{patch} from {doi} does not exist")

    if not os.path.exists(json_file):
        raise RuntimeError(f"{json_file} from {doi} does not exist")

    convert = Convert(json_file, (patch, 0))
    convert.validate_par_names()
    convert.validate_target(rng=RNG)
