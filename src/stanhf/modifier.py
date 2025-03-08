"""
Parse a modifier from a hf model
================================
"""

import warnings
from abc import abstractmethod

import numpy as np

from .stanabc import Stan
from .stanstr import join, add_to_target
from .metadata import add_metadata_comment, add_metadata_entry


class Modifier(Stan):
    """
    Abstract modifier representation
    """

    def __init__(self, modifier, sample):
        self.sample = sample
        self.name = join(sample.name, modifier["type"], modifier["name"])
        self.type = modifier["type"]
        self.par_name = modifier["name"]

    @property
    @abstractmethod
    def par_size(self):
        """
        @returns Size of this modifier parameter
        """

    @property
    @abstractmethod
    def par_init(self):
        """
        @returns Default or initial value of modifier
        """

    @property
    @abstractmethod
    def par_bound(self):
        """
        @returns Sampling bounds for parameter
        """

    @property
    def additive(self):
        """
        @returns Whether modifier is additive (cf. multiplicative)
        """
        return False

    @property
    def is_null(self):
        """
        @returns Whether modifier has no effect
        """
        return False

    @property
    def per_channel(self):
        """
        @returns Whether modifier scope limited to a single channel
        """
        return False

    @property
    def constrained(self):
        """
        @returns Whether modifier is constrained by a term in likelihood
        """
        return False


class Factor(Modifier):
    """
    Scale a sample by a factor
    """
    par_size = 0
    par_init = [1.]
    par_bound = [[0., 10.]]

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Scale a sample by a factor
        """
        return f"{self.sample.par_name} *= {self.par_name};"


class ShapeFactor(Modifier):
    """
    Scale each bin in a sample by different factor
    """
    @property
    def par_bound(self):
        return [[0., 10.]] * self.par_size

    @property
    def par_size(self):
        return self.sample.nbins

    @property
    def par_init(self):
        return [1.] * self.par_size

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Scale the sample by a bin-wise factor
        """
        return f"{self.sample.par_name} .*= {self.par_name};"


class StatError(Modifier):
    """
    Scale each bin in a sample by different factor, but constrain those factors by a normal
    """
    per_channel = True

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.stdev_name = join("stdev", self.name)
        self.stdev = modifier["data"]

    @property
    def par_bound(self):
        return [[0., 10.]] * self.par_size

    @property
    def par_size(self):
        return self.sample.nbins

    @property
    def par_init(self):
        return [1.] * self.par_size

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Scale the sample by a bin-wise factor
        """
        return f"{self.sample.par_name} .*= {self.par_name};"

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare data for standard deviation of normal
        """
        return f"vector[{self.par_size}] {self.stdev_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Set data for standard deviation of normal
        """
        return {self.stdev_name: self.stdev}


class ShapeSys(Modifier):
    """
    Scale each bin in a sample by different factor, but constrain those factors by a Poisson
    representing an auxiliary measurement
    """

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.expected_name = join("expected", self.name)
        self.observed_name = join("observed", self.name)
        self.rel_error_name = join("rel_error", self.name)
        self.rel_error = modifier["data"]

    @property
    def par_bound(self):
        return [[0., 10.]] * self.par_size

    @property
    def par_size(self):
        return self.sample.nbins

    @property
    def par_init(self):
        return [1.] * self.par_size

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Scale the sample by a bin-wise factor and defines rate parameters
        """
        return f"""{self.sample.par_name} .*= {self.par_name};
                   vector[{self.par_size}] {self.expected_name} = {self.par_name} .* {self.observed_name};"""

    @add_metadata_comment
    def stan_trans_data(self):
        """
        @returns Data for auxiliary measurements of rate parameters
        """
        return f"vector[{self.par_size}] {self.observed_name} = square({self.sample.nominal_name} ./ {self.rel_error_name});"

    @add_metadata_comment
    def stan_model(self):
        """
        @returns Poisson constraint for rate parameters
        """
        return add_to_target("poisson_real",
                             self.observed_name, self.expected_name)

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare data for relative error in auxiliary measurement
        """
        return f"vector[{self.par_size}] {self.rel_error_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Set data for standard deviation of normal
        """
        return {self.rel_error_name: self.rel_error}


class HistoSys(Modifier):
    """
    A bin-wise additive modifier from interpolation
    """
    additive = True
    constrained = True
    par_size = 0
    par_init = [0.]
    par_bound = [[-5., 5.]]

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)

        self.lu_data = (modifier["data"]["lo_data"],
                        modifier["data"]["hi_data"])
        self.lu_name = join("lu", self.name)

    @property
    def is_null(self):
        return self.lu_data[0] == self.lu_data[1] == self.sample.nominal

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare one-sigma lower and upper values for additive corrections
        """
        return f"tuple(vector[{self.sample.nbins}], vector[{self.sample.nbins}]) {self.lu_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Set data for one-sigma lower and upper values for additive corrections
        """
        return {self.lu_name: self.lu_data}

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Interpolated additive correction to sample
        """
        return f"{self.sample.par_name} += term_interp({self.par_name}, {self.sample.nominal_name}, {self.lu_name});"


class NormSys(Modifier):
    """
    A multiplicative modifier from interpolation
    """
    constrained = True
    par_size = 0
    par_init = [0.]
    par_bound = [[-5., 5.]]

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.lu_data = (modifier["data"]["lo"], modifier["data"]["hi"])
        self.lu_name = join("lu", self.name)

    @property
    def is_null(self):
        return self.lu_data[0] == self.lu_data[1] == 1.

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare one-sigma lower and upper values for multiplicative corrections
        """
        return f"tuple(real, real) {self.lu_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Set data for one-sigma lower and upper values for multiplicative corrections
        """
        return {self.lu_name: self.lu_data}

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Interpolated multiplicative correction to sample
        """
        return f"{self.sample.par_name} *= factor_interp({self.par_name}, {self.lu_name});"


class StandardNormal(Stan):
    """
    Add a standard normal constraint to a parameter
    """

    def __init__(self, par_name):
        """
        @param param_name Stan parameter constrained by standard normal
        """
        self.par_name = par_name

    @add_metadata_comment
    def stan_model(self):
        """
        @returns Constrain by standard normal
        """
        return add_to_target("std_normal", self.par_name)


class CombinedStatError(Stan):
    """
    Combine statistical errors on a bin from several samples
    """

    def __init__(self, par_name, modifiers, channel):
        self.stdev_name = join("stdev", channel.name, par_name)
        self.par_name = par_name
        self.modifiers = modifiers
        self.channel = channel

        var = sum(np.array(m.stdev)**2 for m in modifiers)
        if np.any(var == 0.):
            warnings.warn(f"variance was zero for some bins for {par_name}")

    @add_metadata_comment
    def stan_trans_data(self):
        """
        @returns Compute standard deviation of measurement
        """
        nominal = " + ".join(m.sample.nominal_name for m in self.modifiers)
        var = " + ".join(f"{m.stdev_name}.^2" for m in self.modifiers)
        return f"vector[{self.channel.nbins}] {self.stdev_name} = sqrt({var}) ./ ({nominal});"

    @add_metadata_comment
    def stan_model(self):
        """
        @returns Constrain by normal centered at one
        """
        return add_to_target("normal", self.par_name, 1., self.stdev_name)


MODIFIERS = {
    "lumi": Factor,
    "normfactor": Factor,
    "shapesys": ShapeSys,
    "histosys": HistoSys,
    "normsys": NormSys,
    "staterror": StatError,
    "shapefactor": ShapeFactor}


def find_constraints(modifiers):
    """
    @returns Find constraints that are applied once to modifiers
    """
    par_name = {m.par_name for m in modifiers if m.constrained and not m.is_null}
    return [StandardNormal(p) for p in par_name]


def find_staterror(channel):
    """
    @returns Combine statistical errors over a channel
    """
    staterror = {}
    for modifier in channel.modifiers:
        if modifier.type == "staterror":
            staterror.setdefault(modifier.par_name, [])
            staterror[modifier.par_name].append(modifier)

    return [CombinedStatError(k, v, channel) for k, v in staterror.items()]


def find_modifier(modifier, *args, **kwargs):
    """
    @returns Modifier from hf modifier data
    """
    cls = MODIFIERS[modifier["type"]]
    return cls(modifier, *args, **kwargs)


def order_modifiers(modifiers):
    """
    @returns Modifiers, but with additive modifiers applied first
    """
    add = [m for m in modifiers if m.additive]
    other = [m for m in modifiers if not m.additive]
    return add + other


def check_per_channel(modifiers):
    """
    Check that per channel modifiers are not repeated across channels
    """
    per_channel_modifiers = [m for m in modifiers if m.per_channel]
    channels = {m.par_name: m.sample.channel.name for m in per_channel_modifiers}

    for m in per_channel_modifiers:
        if m.sample.channel.name != channels[m.par_name]:
            raise RuntimeError(
                f"The {m.type} modifier scope is per channel - "
                f"repeated {m.par_name} across channels")
