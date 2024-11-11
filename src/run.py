"""
Running and validating converted models
=======================================
"""

import json

import numpy as np
import pyhf
from cmdstanpy import CmdStanModel

from .stanstr import flatten, squeeze
from .tracer import METADATA


def builder(root, **kwargs):
    """
    @param root Root name for Stan files
    @returns Stan model
    """
    return CmdStanModel(stan_file=f"{root}.stan", **kwargs)


class StanHf:
    """
    Wrapper for Stan implementation of hf
    """

    def __init__(self, root):
        """
        @param root Root name for Stan files
        """
        self.model = builder(root)
        self.data = f"{root}_data.json"

    def par_names(self):
        return list(self.model.src_info()['parameters'].keys())

    def target(self, pars):
        data_frame = self.model.log_prob(pars,
                                         data=self.data,
                                         jacobian=False, sig_figs=18)
        return data_frame["lp__"].values[0]


class NativeHf:
    """
    Wrapper for native pyhf
    """

    def __init__(self, root):
        """
        @param root Root name for Stan files
        """
        with open(f"{root}.json", encoding="utf-8") as hf_file:
            spec = json.load(hf_file)

        workspace = pyhf.Workspace(spec)
        self.model = workspace.model()
        self.data = workspace.data(self.model)

    def par_names(self):
        return self.model.config.parameters

    def target(self, pars):
        pars = flatten([pars[k] for k in self.par_names()])
        return self.model.logpdf(pars, self.data)[0]


def perturb(x, scale=0.01, rng=None):
    """
    @param x Parameter to be pertubed
    @returns Parameter perturbed by random standard normal deviate
    """
    if rng is None:
        rng = np.random.default_rng()

    p = x + scale * rng.standard_normal(np.size(x))
    return squeeze(p)

def validator(root, rng=None):
    """
    @param root Root name for Stan files
    @raises if disagreement between target in Stan and pyhf
    """
    stanhf = StanHf(root)
    nhf = NativeHf(root)

    stanhf_par_names = set(stanhf.par_names())
    nhf_par_names = set(nhf.par_names())

    if stanhf_par_names != nhf_par_names:
        raise RuntimeError(
            f"no agreement in parameter names: Stan = {stanhf_par_names} vs. pyhf = {nhf_par_names}")

    with open(f'{root}_init.json', encoding="utf-8") as init_file:
        pars = json.load(init_file)

    pars.pop(METADATA)

    for k in pars:
        pars[k] = perturb(pars[k], rng=rng)

    stanhf_target = stanhf.target(pars)
    nhf_target = nhf.target(pars)

    if not np.isclose(stanhf_target, nhf_target):
        raise RuntimeError(
            f"no agreement in target: Stan = {stanhf_target} vs. pyhf = {nhf_target} for pars = {pars}")
