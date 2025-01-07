"""
Test conversion of big model
============================
"""

import glob
import os

import pytest
import pyhf.contrib.utils

from stanhf import convert


DOI = ["10.17182/hepdata.134244",
       "10.17182/hepdata.95928",
       "10.17182/hepdata.133620",
       "10.17182/hepdata.129959",
       "10.17182/hepdata.116034",
       "10.17182/hepdata.105864",
       "10.17182/hepdata.104458",
       "10.17182/hepdata.105039",
       "10.17182/hepdata.104860",
       "10.17182/hepdata.95751",
       "10.17182/hepdata.100351",
       "10.17182/hepdata.100174",
       "10.17182/hepdata.97041",
       "10.17182/hepdata.99806",
       "10.17182/hepdata.98796",
       "10.17182/hepdata.95664",
       "10.17182/hepdata.100170",
       "10.17182/hepdata.95748",
       "10.17182/hepdata.91760",
       "10.17182/hepdata.91127",
       "10.17182/hepdata.91374",
       "10.17182/hepdata.92006",
       "10.17182/hepdata.90607.v2",
       "10.17182/hepdata.91214.v3",
       "10.17182/hepdata.89413",
       "10.17182/hepdata.89408"]


def runner(doi):
    """
    Convert hep-data model
    """
    folder_name = doi.replace("/", "_").replace(".", "_")
    pyhf.contrib.utils.download(f"https://doi.org/{doi}", folder_name)
    for json_file in glob.glob(f"{folder_name}/*.json"):
        if "patch" not in json_file:
            convert(json_file)


def test_convert():
    """
    Test one data set
    """
    runner("10.17182/hepdata.90607.v3/r3")


@pytest.mark.skipif(
    not os.environ.get("PYTEST_RUN_SLOW"),
    reason="PYTEST_RUN_SLOW not set in environment"
)
@pytest.mark.parametrize("doi", DOI)
def test_thorough_convert(doi):
    """
    Test all data sets
    """
    runner(doi)
