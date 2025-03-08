"""
Test conversion of big model
============================
"""

import os

import numpy as np
import pytest
import pyhf.contrib.utils

from stanhf import Convert


RNG = np.random.default_rng(111)


DATA = [("10.17182/hepdata.134244.v1/r2", "BkgOnly.json"),
        ("10.17182/hepdata.95928", "BkgOnly.json"),
        ("10.17182/hepdata.133620", "BkgOnly.json"),
        ("10.17182/hepdata.129959", "BkgOnly.json"),
        ("10.17182/hepdata.116034", "BkgOnly.json"),
        ("10.17182/hepdata.105864", "BkgOnly.json"),
        ("10.17182/hepdata.104458", "BkgOnly.json"),
        ("10.17182/hepdata.105039", "BkgOnly.json"),
        ("10.17182/hepdata.104860", "BkgOnly.json"),
        ("10.17182/hepdata.95751", "BkgOnly.json"),
        ("10.17182/hepdata.100351", "BkgOnly.json"),
        ("10.17182/hepdata.100174", "BkgOnly.json"),
        ("10.17182/hepdata.97041", "BkgOnly.json"),
        ("10.17182/hepdata.99806", "BkgOnly.json"),
        ("10.17182/hepdata.98796", "BkgOnly.json"),
        ("10.17182/hepdata.95664", "BkgOnly.json"),
        ("10.17182/hepdata.100170", "BkgOnly.json"),
        ("10.17182/hepdata.95748", "BkgOnly.json"),
        ("10.17182/hepdata.91760", "BkgOnly.json"),
        ("10.17182/hepdata.91127", "BkgOnly.json"),
        ("10.17182/hepdata.91374", "BkgOnly.json"),
        ("10.17182/hepdata.92006", "BkgOnly.json"),
        ("10.17182/hepdata.90607.v2", "BkgOnly.json"),
        ("10.17182/hepdata.91214.v3", "BkgOnly.json"),
        ("10.17182/hepdata.89413", "BkgOnly.json"),
        ("10.17182/hepdata.89408.v3/r2", "RegionA/BkgOnly.json"),
        ("10.17182/hepdata.89408.v3/r2", "RegionB/BkgOnly.json"),
        ("10.17182/hepdata.89408.v3/r2", "RegionC/BkgOnly.json")]


def fetch_hep_data(doi):
    """
    @returns Downloaded json file path
    """
    folder_name = doi.replace("/", "_").replace(".", "_")

    if not os.path.exists(folder_name):
        url = f"https://doi.org/{doi}"
        print(f"downloading from {url} to {folder_name}")
        pyhf.contrib.utils.download(url, folder_name)
        print(f"downloaded {folder_name}")

    return folder_name


@pytest.mark.parametrize("data", DATA if os.environ.get("PYTEST_ALL_HEP_DATA") else DATA[-3:])
def test_validate_hep_data(data):
    """
    Test a HEP-DATA histfactory model
    """
    doi, json_file = data
    folder_name = fetch_hep_data(doi)
    path = os.path.join(folder_name, json_file)

    if not os.path.exists(path):
        raise RuntimeError(f"{path} from {doi} does not exist")

    convert = Convert(path)
    convert.validate_par_names()
    convert.validate_target(rng=RNG)
