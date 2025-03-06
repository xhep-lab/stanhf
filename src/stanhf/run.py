"""
Running pyhf and stanhf models
==============================
"""

import json
import warnings

import numpy as np
from cmdstanpy import CmdStanModel, install_cmdstan, cmdstan_path

from .pars import get_pyhf_pars
from .metadata import METADATA


def install(progress=True, **kwargs):
    """
    Install Stan if it doesn't exist already
    """
    try:
        return cmdstan_path()
    except ValueError as error:
        warnings.warn(f"Stan not found --- {error}. Installing it.")
        install_cmdstan(progress=progress, **kwargs)
        return cmdstan_path()


def run_stanhf_model(pars, data_file_name, exe_file_name):
    """
    Run stanhf model on a particular point
    """
    model = CmdStanModel(exe_file=exe_file_name)
    data_frame = model.log_prob(pars,
                                data=data_file_name,
                                jacobian=False, sig_figs=18)
    return data_frame["lp__"].values[0]


def run_pyhf_model(pars, workspace):
    """
    Run pyhf model on a particular point
    """
    model = workspace.model(poi_name=None)
    data = workspace.data(model)
    return model.logpdf(get_pyhf_pars(pars, model), data)[0]


def perturb(param, scale=0.01, rng=None):
    """
    @param param Parameter to be perturbed
    @returns Parameter perturbed by random normal deviate
    """
    if rng is None:
        rng = np.random.default_rng()

    pertubed = param + scale * rng.standard_normal(np.size(param))
    return pertubed.reshape(np.shape(param)).tolist()


def perturb_param_file(param_file_name, rng=None):
    """
    @returns Initial parameters perturbed by random noise
    """
    with open(param_file_name, encoding="utf-8") as param_file:
        pars = json.load(param_file)
    return {k: perturb(pars[k], rng=rng) for k in pars if k != METADATA}
