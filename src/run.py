"""
Running and validating converted models
=======================================
"""

import json

import numpy as np
import pyhf
from cmdstanpy import CmdStanModel

from .stanstr import flatten


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

        with open(f'{root}_init.json', encoding="utf-8") as init_file:
            self.pars = json.load(init_file)

    def par_names(self):
        return list(self.model.src_info()['parameters'].keys())

    def target(self):
        data_frame = self.model.log_prob(self.pars,
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
        with open(f'{root}_init.json', encoding="utf-8") as init_file:
            self.pars = json.load(init_file)

        with open(f"{root}.json", encoding="utf-8") as hf_file:
            spec = json.load(hf_file)

        workspace = pyhf.Workspace(spec)
        self.model = workspace.model()
        self.data = workspace.data(self.model)

    def par_names(self):
        return self.model.config.parameters

    def target(self):
        pars = flatten([self.pars[k] for k in self.par_names()])
        return self.model.logpdf(pars, self.data)[0]


def validator(root):
    """
    @param root Root name for Stan files
    @raises if disagreement between target in Stan and pyhf
    """
    stanhf = StanHf(root)
    nhf = NativeHf(root)

    if not np.isclose(stanhf.target(), nhf.target()):
        raise RuntimeError(
            f"no agreement in target: Stan = {stanhf} vs. pyhf = {nhf}")

    if set(stanhf.par_names()) != set(nhf.par_names()):
        raise RuntimeError(
            f"no agreement in parameter names: Stan = {stanhf} vs. pyhf = {nhf}")
