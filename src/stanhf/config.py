"""
Parse configuration from a hf model
===================================

Including measurements, initial choices of parameters and bounds.
"""

import warnings

from .stanabc import Stan
from .stanstr import join, add_to_target, read_par_bound, read_par_init
from .metadata import add_metadata_comment, add_metadata_entry


class Measured(Stan):
    """
    Normal measurement of a modifier parameter
    """

    def __init__(self, config):
        """
        @param config hf configuration data
        """
        self.par_name = config["name"]
        self.normal_data_name = join("normal", self.par_name)
        self.normal_data = tuple(config[k][0] for k in ["auxdata", "sigmas"])

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare data for normal log-likelihood
        """
        return f"tuple(real, real) {self.normal_data_name};"

    @add_metadata_comment
    def stan_model(self):
        """
        @returns Normal log-likelihood for modifier
        """
        return add_to_target("normal", self.par_name,
                             f"{self.normal_data_name}.1", f"{self.normal_data_name}.2")

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Data for normal log-likelihood
        """
        return {self.normal_data_name: self.normal_data}


class FreeParameter(Stan):
    """
    Declare a parameter that is sampled
    """

    def __init__(self, par_name, par_size, par_init, par_bound):
        """
        @param par_name Name of parameter
        """
        self.par_name = par_name
        self.par_size = par_size
        self.par_init = par_init
        self.par_bound = par_bound
        self.par_bound_name = join("lu", self.par_name)

    @add_metadata_comment
    def stan_pars(self):
        """
        @returns Declare parameter
        """
        bound = f"<lower={self.par_bound_name}.1, upper={self.par_bound_name}.2>"
        if self.par_size == 0:
            return f"real{bound} {self.par_name};"
        return f"vector{bound}[{self.par_size}] {self.par_name};"

    @add_metadata_entry
    def stan_init_card(self):
        """
        @returns Initialization or default for parameter
        """
        return {self.par_name: self.par_init}

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare lower and upper bound for parameter
        """
        if self.par_size == 0:
            return f"tuple(real, real) {self.par_bound_name};"
        return f"tuple(vector[{self.par_size}], vector[{self.par_size}]) {self.par_bound_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Data for bounds for parameter
        """
        return {self.par_bound_name: self.par_bound}


class POI(Stan):
    """
    Declare a parameter that is the parameter of interest
    """

    def __init__(self, par_name, par_size, par_init, par_bound):
        """
        @param par_name Name of parameter
        """
        self.par_name = par_name
        self.par_size = par_size

        if self.par_size != 0:
            raise RuntimeError("POI must be a scalar")

        self.par_init = par_init
        self.par_bound = par_bound
        self.par_bound_name = join("lu", self.par_name)
        self.free_par_name = join("free", self.par_name)
        self.fixed_par_name = join("fixed", self.par_name)
        self.fix_flag = join("fix", self.par_name)

    @add_metadata_comment
    def stan_pars(self):
        """
        @returns Declare parameter
        """
        bound = f"<lower={self.par_bound_name}.1, upper={self.par_bound_name}.2>"
        return f"array[1 - {self.fix_flag}] real{bound} {self.free_par_name};"

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Declare parameter
        """
        return f"real {self.par_name} = {self.fix_flag} ? {self.fixed_par_name} : {self.free_par_name}[1];"

    @add_metadata_entry
    def stan_init_card(self):
        """
        @returns Initialization or default for parameter
        """
        return {self.free_par_name: [self.par_init]}

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare lower and upper bound for parameter
        """
        return f"""
                int<lower=0, upper=1> {self.fix_flag};
                real {self.fixed_par_name};
                tuple(real, real) {self.par_bound_name};
                """

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Data for bounds for parameter
        """
        return {self.par_bound_name: self.par_bound, self.fix_flag: False, self.fixed_par_name: self.par_init}


class FixedParameter(Stan):
    """
    Declare a fixed parameter
    """

    def __init__(self, par_name, par_size, par_init):
        """
        @param par_name Name of parameter
        """
        self.par_name = par_name
        self.par_size = par_size
        self.par_init = par_init

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare lower and upper bound for parameter
        """
        if self.par_size == 0:
            return f"real {self.par_name};"
        return f"vector[{self.par_size}] {self.par_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Fixed value for parameter
        """
        return {self.par_name: self.par_init}


class NullParameter(Stan):
    """
    A null parameter
    """

    def __init__(self, par_name, par_size):
        """
        @param par_name Name of parameter
        """
        self.par_name = par_name
        self.par_size = par_size


def is_measured(config, par_name):
    """
    @returns Whether configuration indicates a measurement
    """
    return {"auxdata", "sigmas"} <= config.get(par_name, {}).keys()


def find_measureds(config, modifiers):
    """
    @returns Measurement objects
    """
    unique = {m.par_name for m in modifiers if not m.is_null}
    return [Measured(config.get(p, {})) for p in unique if is_measured(config, p)]


def find_par_prop(modifiers, prop):
    """
    @returns Parameter name from modifiers
    """
    d = [getattr(m, prop) for m in modifiers]
    for e in d:
        if e != d[0]:
            warnings.warn(f"modifiers have inconsistent {prop}: {d}")
    return d.pop()


def find_param(poi, par_name, par_config, modifiers):
    """
    @returns Parameter bounds from modifiers
    """
    par_size = find_par_prop(modifiers, "par_size")

    par_init = par_config.get("inits", find_par_prop(modifiers, "par_init"))
    par_init = read_par_init(par_init, par_size)

    par_bound = par_config.get("bounds", find_par_prop(modifiers, "par_bound"))
    par_bound = read_par_bound(par_bound, par_size)

    if all(m.is_null for m in modifiers):
        return NullParameter(par_name, par_size)

    if par_config.get("fixed"):
        return FixedParameter(par_name, par_size, par_init)

    if par_name == poi:
        return POI(par_name, par_size, par_init, par_bound)

    return FreeParameter(par_name, par_size, par_init, par_bound)


def find_params(poi, config, modifiers):
    """
    @returns Parameters from data in configuration and hf model
    """
    groups = {m.par_name: [
        l for l in modifiers if l.par_name == m.par_name] for m in modifiers}
    return [find_param(poi, p, config.get(p, {}), m) for p, m in groups.items()]
