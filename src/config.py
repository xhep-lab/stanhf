"""
Parse configuaration from a hf model
====================================

Including measurements, initial choices of parameters and bounds.
"""

from .stanabc import Stan
from .stanstr import join, add_to_target, squeeze
from .tracer import trace


class Measured(Stan):
    """
    Normal measurement of a modifier paramter
    """

    def __init__(self, config):
        """
        @param config hf configuaration data
        """
        self.par_name = config['name']
        self.data_names = [join(k, self.par_name)
                           for k in ["mu", "sigma"]]
        self.data = [squeeze(config[k]) for k in ["auxdata", "sigmas"]]

    @trace
    def stan_data(self):
        """
        @returns Declare data for normal log-likelihood
        """
        return f"""real {self.data_names[0]};
                   real {self.data_names[1]};"""

    @trace
    def stan_model(self):
        """
        @returns Normal log-likelihood for modifier
        """
        return add_to_target(
            "normal_lpdf", self.par_name, *self.data_names)

    @trace
    def stan_data_card(self):
        """
        @returns Data for normal log-likelihood
        """
        return dict(zip(self.data_names, self.data))


class Parameter(Stan):
    """
    Declare a parameter
    """

    def __init__(self, modifier, config):
        """
        @param par_name Stan variable name of parameter
        @param par_size Size of parameter e.g. could be vector
        @param fixed Whether parameter should be fixed
        @param bounds Tuple of upper and lower bounds for parameter
        """
        self.par_name = modifier.par_name
        self.par_size = modifier.par_size
        self.par_init = config.get("init", modifier.par_init)
        self.par_fix = config.get("fixed", False)
        self.par_bound = squeeze(config.get("bounds", None))

    @property
    def bound_str(self):
        """
        @returns Stan string for variable bounds, e.g. <lower=0, upper=1>
        """
        if self.par_bound is None:
            return str()
        return f"<lower={self.par_bound[0]}, upper={self.par_bound[1]}>"

    @property
    def type_str(self):
        """
        @returns Stan string for variable type, e.g., real
        """
        if self.par_size == 1:
            return f'real{self.bound_str}'
        return f"vector{self.bound_str}[{self.par_size}]"

    @trace
    def stan_pars(self):
        """
        @returns Declare parameter, if it isn't fixed
        """
        if self.par_fix:
            return None
        return f"{self.type_str} {self.par_name};"

    @trace
    def stan_trans_data(self):
        """
        @returns Set parameter, if it is fixed
        """
        if not self.par_fix:
            return None
        return f"{self.type_str} {self.par_name} = {self.par_init};"

    @trace
    def stan_init_card(self):
        """
        @returns Initialization or default for parameter
        """
        return {self.par_name: self.par_init}


def find_measureds(config):
    """
    @returns Measured modifiers from configuration
    """
    return [Measured(d)
            for d in config.values() if {'auxdata', 'sigmas'} <= d.keys()]


def find_params(config, modifiers):
    """
    @returns Parameters from data in configuation and hf model
    """
    unique = {m.par_name: m for m in modifiers}
    return [Parameter(v, config.get(k, {})) for k, v in unique.items()]
