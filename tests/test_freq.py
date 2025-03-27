"""
Test Stan interface to pyhf
===========================
"""

import json
import numpy as np
import os

import pyhf

from stanhf import Convert
from stanhf.contrib.freq import MockPyhfModel, mock_pyhf_backend


CWD = os.path.dirname(os.path.realpath(__file__))
EXAMPLE = os.path.normpath(os.path.join(CWD, "..", "examples", "model.json"))


def nhf_upper_limits(json_file):
    """
    @returns Upper limits from pyhf
    """
    with open(json_file, encoding="utf-8") as f:
        workspace = pyhf.Workspace(json.load(f))

    model = workspace.model()
    data = workspace.data(model)

    obs_limit, exp_limits, _ = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, level=0.05, return_results=True)

    return obs_limit, exp_limits


def shf_upper_limits(json_file):
    """
    @returns Upper limits from Stan interface to pyhf
    """
    convert = Convert(json_file)
    stan_file_name, data_file_name, init_file_name = convert.write_to_disk()
    model = MockPyhfModel(stan_file_name, data_file_name, init_file_name)

    with mock_pyhf_backend():
        obs_limit, exp_limits, _ = pyhf.infer.intervals.upper_limits.upper_limit(
            model.config.data, model, level=0.05, return_results=True
        )

    return obs_limit, exp_limits


def test_freq():
    """
    Compare native pyhf results to those using Stan interface
    """

    nhf_obs_limit, nhf_exp_limits = nhf_upper_limits(EXAMPLE)
    shf_obs_limit, shf_exp_limits = shf_upper_limits(EXAMPLE)

    assert np.isclose(nhf_obs_limit, shf_obs_limit, rtol=1e-3)
    assert np.all(np.isclose(nhf_exp_limits, shf_exp_limits, rtol=1e-3))
