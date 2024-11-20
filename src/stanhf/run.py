"""
Running and validating converted models
=======================================
"""

import json
import warnings

import numpy as np
import pyhf
from cmdstanpy import CmdStanModel, compile_stan_file, install_cmdstan, cmdstan_path

from .stanstr import flatten
from .tracer import METADATA



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


def build(root, cxx="clang++", cxx_optim_level=0, **kwargs):
    """
    @param root Root name for Stan files
    """
    compile_stan_file(
        f"{root}.stan",
        cpp_options={
            "O": cxx_optim_level,
            "CXX": cxx},
        **kwargs)


class StanHf:
    """
    Wrapper for Stan implementation of hf
    """

    def __init__(self, root):
        """
        @param root Root name for Stan files
        """
        self.model = CmdStanModel(stan_file=f"{root}.stan", exe_file=root)
        self.data = f"{root}_data.json"

    def par_names(self):
        """
        @returns Arbitrary order parameter names
        """
        return list(self.model.src_info()["parameters"].keys())

    def target(self, pars):
        """
        @returns Stan target function
        """
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
        """
        @returns Ordered parameter names
        """
        return self.model.config.par_order

    def target(self, pars):
        """
        @returns pyhf target function
        """
        pars = flatten([pars[k] for k in self.par_names()])
        return self.model.logpdf(pars, self.data)[0]


def perturb(param, scale=0.01, rng=None):
    """
    @param x Parameter to be pertubed
    @returns Parameter perturbed by random normal deviate
    """
    if rng is None:
        rng = np.random.default_rng()

    pertubed = param + scale * rng.standard_normal(np.size(param))
    return pertubed.reshape(np.shape(param)).tolist()


def validate(root, rng=None):
    """
    @param root Root name for model files that will be checked
    """
    stanhf = StanHf(root)
    nhf = NativeHf(root)

    stanhf_par_names = set(stanhf.par_names())
    nhf_par_names = set(nhf.par_names())

    if stanhf_par_names != nhf_par_names:
        raise RuntimeError(
             "no agreement in parameter names: "
            f"Stan = {stanhf_par_names} vs. pyhf = {nhf_par_names}")

    with open(f"{root}_init.json", encoding="utf-8") as init_file:
        pars = json.load(init_file)

    pars.pop(METADATA)

    for k in pars:
        pars[k] = perturb(pars[k], rng=rng)

    stanhf_target = stanhf.target(pars)
    nhf_target = nhf.target(pars)

    if not np.isclose(stanhf_target, nhf_target):
        raise RuntimeError(
             "no agreement in target: "
            f"Stan = {stanhf_target} vs. pyhf = {nhf_target} for pars = {pars}")
