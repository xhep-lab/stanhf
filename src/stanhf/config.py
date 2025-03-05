"""
Parse configuaration from a hf model
====================================

Including measurements, initial choices of parameters and bounds.
"""

from .stanabc import Stan
from .stanstr import join, add_to_target, read_par_bound, read_par_init
from .tracer import add_metadata_comment, add_metadata_entry
from .modifier import CONSTRAINED


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
        return add_to_target("normal_lpdf", self.par_name,
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


class FixedParameter(Stan):
    """
    Declare a fixed parameter
    """

    def __init__(self, config, modifier):
        """
        @param config hf configuration for this parameter
        @param modifier Modifier object for this parameter
        """
        self.par_name = modifier.par_name
        self.par_size = modifier.par_size
        par_init = config.get("inits", modifier.par_init)
        self.par_init = read_par_init(par_init, self.par_size)

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

    def __init__(self, modifier):
        """
        @param config hf configuration for this parameter
        @param modifier Modifier object for this parameter
        """
        self.par_name = modifier.par_name
        self.par_size = modifier.par_size


def is_measured(config, modifier):
    """
    @returns Whether configuration indicates a measurement
    """
    return {"auxdata", "sigmas"} <= config.get(modifier.par_name, {}).keys()


def find_measureds(config, modifiers):
    """
    @returns Measured modifiers
    """
    unique = {m.par_name: m for m in modifiers}.values()
    return [Measured(config.get(m.par_name, {})) for m in unique if is_measured(config, m)]


def find_param(config, modifier):
    """
    @returns Parameter from modifier
    """
    config_data = config.get(modifier.par_name, {})

    if config_data.get("fixed"):
        return FixedParameter(config_data, modifier)

    if modifier.is_null:
        if is_measured(config, modifier) or modifier.type in CONSTRAINED:
            return FixedParameter(config_data, modifier)
        return NullParameter(modifier)

    return FreeParameter(config_data, modifier)


def find_params(config, modifiers):
    """
    @returns Parameters from data in configuation and hf model
    """
    return [find_param(config, m) for m in modifiers]
