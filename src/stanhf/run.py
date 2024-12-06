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


def build(root, **kwargs):
    """
    @param root Root name for Stan files
    """
    compile_stan_file(f"{root}.stan", **kwargs)


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

    def fixed_par_names(self):
        """
        @returns Names of fixed parameters
        """
        fixed = self.model.config.suggested_fixed()
        return [n for f, n in zip(fixed, self.par_names()) if f]

    def par_names(self):
        """
        @returns Ordered parameter names
        """
        return self.model.config.par_order

    def target(self, pars):
        """
        @returns pyhf target function
        """
        inits = self.model.config.suggested_init()
        pars = flatten([pars.get(k, inits[i])
                       for i, k in enumerate(self.par_names())])
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


def validate(root, par, fixed, null, rng=None):
    """
    @param root Root name for model files that will be checked
    """
    stanhf = StanHf(root)
    nhf = NativeHf(root)

    nhf_fixed = nhf.fixed_par_names()

    if set(fixed) != set(nhf_fixed):
        raise RuntimeError(
            "no agreement in fixed parameter names: "
            f"Stan = {fixed} [{len(fixed)}]"
            f" vs. pyhf = {nhf_fixed} [{len(nhf_fixed)}]")

    stanhf_par = stanhf.par_names()

    if set(stanhf_par) != set(par):
        raise RuntimeError(
            "no agreement in parameter names: "
            f"Stan = {stanhf_par} [{len(stanhf_par)}]"
            f" vs. expected = {par} [{len(par)}]")

    nhf_par = nhf.par_names()

    if set(par + fixed + null) != set(nhf_par):
        raise RuntimeError(
            "no agreement in parameter names: "
            f"Stan = {stanhf_par} [{len(stanhf_par)}]"
            f" vs. pyhf = {nhf_pars} [{len(nhf_par)}]")

    with open(f"{root}_init.json", encoding="utf-8") as init_file:
        pars = json.load(init_file)

    pars.pop(METADATA)

    for k in pars:
        pars[k] = perturb(pars[k], rng=rng)

    stanhf_target = stanhf.target(pars)
    nhf_target = nhf.target(pars)

    if not np.isclose(stanhf_target, nhf_target):
        raise RuntimeError(
            f"no agreement in target: Stan = {stanhf_target}"
            f" vs. pyhf = {nhf_target} for pars = {pars}")
