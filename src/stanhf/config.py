"""
Parse configuaration from a hf model
====================================

Including measurements, initial choices of parameters and bounds.
"""

from abc import abstractmethod

from .stanabc import Stan
from .stanstr import join, add_to_target, read_par_bound, read_par_init
from .tracer import trace


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

    @trace
    def stan_data(self):
        """
        @returns Declare data for normal log-likelihood
        """
        return f"tuple(real, real) {self.normal_data_name};"

    @trace
    def stan_model(self):
        """
        @returns Normal log-likelihood for modifier
        """
        return add_to_target("normal_lpdf", self.par_name,
                             f"{self.normal_data_name}.1", f"{self.normal_data_name}.2")

    @trace
    def stan_data_card(self):
        """
        @returns Data for normal log-likelihood
        """
        return {self.normal_data_name: self.normal_data}


class Parameter(Stan):
    """
    Abstract representation of a parameter
    """
    @property
    @abstractmethod
    def par_fixed(self):
        """
        @returns Whether parameter was fixed
        """


class FreeParameter(Parameter):
    """
    Declare a parameter that is sampled
    """
    par_fixed = False

    def __init__(self, config, modifier):
        """
        @param config hf configuration for this parameter
        @param modifier Modifier object for this parameter
        """
        self.par_name = modifier.par_name
        self.par_size = modifier.par_size
        par_init = config.get("inits", modifier.par_init)
        self.par_init = read_par_init(par_init, self.par_size)
        par_bound = config.get("bounds", modifier.par_bound)
        self.par_bound = read_par_bound(par_bound, self.par_size)
        self.par_bound_name = join("lu", self.par_name)

    @trace
    def stan_pars(self):
        """
        @returns Declare parameter
        """
        bound = f"<lower={self.par_bound_name}.1, upper={self.par_bound_name}.2>"
        if self.par_size == 0:
            return f"real{bound} {self.par_name};"
        return f"vector{bound}[{self.par_size}] {self.par_name};"

    @trace
    def stan_init_card(self):
        """
        @returns Initialization or default for parameter
        """
        return {self.par_name: self.par_init}

    @trace
    def stan_data(self):
        """
        @returns Declare lower and upper bound for parameter
        """
        if self.par_size == 0:
            return f"tuple(real, real) {self.par_bound_name};"
        return f"tuple(vector[{self.par_size}], vector[{self.par_size}]) {self.par_bound_name};"

    @trace
    def stan_data_card(self):
        """
        @returns Data for bounds for parameter
        """
        return {self.par_bound_name: self.par_bound}


class FixedParameter(Parameter):
    """
    Declare a fixed parameter
    """
    par_fixed = True

    def __init__(self, config, modifier):
        """
        @param config hf configuration for this parameter
        @param modifier Modifier object for this parameter
        """
        self.par_name = modifier.par_name
        self.par_size = modifier.par_size
        par_init = config.get("inits", modifier.par_init)
        self.par_init = read_par_init(par_init, self.par_size)

    @trace
    def stan_data(self):
        """
        @returns Declare lower and upper bound for parameter
        """
        if self.par_size == 0:
            return f"real {self.par_name};"
        return f"vector[{self.par_size}] {self.par_name}"

    @trace
    def stan_data_card(self):
        """
        @returns Fixed value for parameter
        """
        return {self.par_name: self.par_init}


def is_measured(config, par_names):
    """
    @returns Whether configuration indicates a measurement
    """
    return {"auxdata", "sigmas"} <= config.keys() and config["name"] in par_names


def find_measureds(config, modifiers):
    """
    @returns Measured modifiers
    """
    par_names = {m.par_name for m in modifiers}
    return [Measured(d) for d in config.values() if is_measured(d, par_names)]


def find_param(config, modifier):
    """
    @returns Parameter or fixed parameter
    """
    if config.get("fixed"):
        return FixedParameter(config, modifier)
    return FreeParameter(config, modifier)


def find_params(config, modifiers):
    """
    @returns Parameters from data in configuation and hf model
    """
    unique = {m.par_name: m for m in modifiers}
    return [find_param(config.get(k, {}), v) for k, v in unique.items()]
