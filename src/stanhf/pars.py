"""
Deal with parameter naming conventions across Stan and pyhf
===========================================================
"""

from cmdstanpy import compilation

from .stanstr import flatten, remove_prefix


def get_pyhf_pars(pars, model):
    """
    @returns pyhf parameters for calling target
    """
    init = get_pyhf_init(model)
    strip_free_pars = {remove_prefix(k, "free_"): v for k, v in pars.items()}
    pars = {k: strip_free_pars.get(k, v) for k, v in init.items()}
    return flatten(pars[k] for k in model.config.par_order)


def get_pyhf_init(model):
    """
    @returns Suggested initial values for parameters
    """
    init = model.config.suggested_init()
    return {p: init[model.config.par_slice(p)] for p in model.config.par_order}


def get_pyhf_par_data(workspace):
    """
    @returns Names and sizes of pyhf parameters
    """
    model = workspace.model(poi_name=None)
    init = get_pyhf_init(model)
    return {k: len(v) for k, v in init.items()}


def get_stan_par_names(stan_file_name):
    """
    @returns Names of Stan model parameters
    """
    src_info = compilation.src_info(
        stan_file_name, compilation.CompilerOptions())
    return list(src_info["parameters"].keys())
