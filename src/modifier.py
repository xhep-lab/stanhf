"""
Parse a modifier from a hf model
================================
"""

from abc import abstractmethod

from .stanabc import Stan
from .stanstr import join, add_to_target, squeeze
from .tracer import trace


class Modifier(Stan):
    """
    Abstract modifier representation.
    """

    def __init__(self, modifier, sample):
        self.modifier = modifier
        self.sample = sample
        self.par_name = self.modifier['name']

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


class Factor(Modifier):
    """
    Scale a sample by a factor
    """
    par_size = 1
    par_init = 1

    @trace
    def stan_trans_pars(self):
        """
        @returns Scales sample by a factor
        """
        return f"{self.sample.par_name} *= {self.par_name};"


class BinWiseFactor(Modifier):
    """
    Scale each bin in a sample by different factor
    """
    @property
    def par_size(self):
        return self.sample.nbins

    @property
    def par_init(self):
        return [1] * self.par_size

    @trace
    def stan_trans_pars(self):
        """
        @returns Scale the sample by a bin-wise factor
        """
        return f"{self.sample.par_name} .*= {self.par_name};"


class NormalBinWiseFactor(BinWiseFactor):
    """
    Scale each bin in a sample by different factor, but constrain those factors by a normal
    """

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.data_name = join("x", self.par_name)
        self.data = squeeze(self.modifier["data"])

    @trace
    def stan_model(self):
        """
        @returns Constrain each bin-wise factor by a normal
        """
        return add_to_target(
            "normal_lpdf", self.par_name, 1, self.data_name)

    @trace
    def stan_data(self):
        """
        @returns Declare data for standard deviation of normal
        """
        return f'vector[{self.par_size}] {self.data_name};'

    @trace
    def stan_data_card(self):
        """
        @returns Set data for standard deviation of normal
        """
        return {self.data_name: self.data}


class PoissonBinWiseFactor(NormalBinWiseFactor):
    """
    Scale each bin in a sample by different factor, but constrain those factors by a Poisson
    """

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.rate_name = join('lambda', self.par_name)
        self.aux_name = join('naux', self.par_name)

    @trace
    def stan_trans_pars(self):
        """
        @returns Scale the sample by a bin-wise factor and defines rate parameters
        """
        return f"""{self.sample.par_name} .*= {self.par_name};
                   vector<lower=0>[{self.par_size}] {self.rate_name} = {self.par_name} .* {self.aux_name};"""

    @trace
    def stan_trans_data(self):
        """
        @returns Data for auxilliary measurements of rate parameters
        """
        return f"vector<lower=0>[{self.par_size}] {self.aux_name} = square({self.sample.data_name} ./ {self.data_name});"

    @trace
    def stan_model(self):
        """
        @returns Poisson constrained for rate parameters
        """
        return add_to_target("poisson_real_lpdf",
                             self.aux_name, self.rate_name)


class CorrelatedBinWiseAdditive(Modifier):
    """
    A bin-wise additive modifier. The additive corrections are correlated
    """
    par_size = 1
    par_init = 0

    def __init__(self, modifier, sample):
        super().__init__(modifier, sample)
        self.lo_data_name = join("x", "lo", self.par_name)
        self.hi_data_name = join("x", "hi", self.par_name)
        self.lo_data = squeeze(modifier["data"]["lo_data"])
        self.hi_data = squeeze(modifier["data"]["hi_data"])

    @trace
    def stan_data(self):
        """
        @returns Declare one-sigma lower and upper values for additive corrections
        """
        return f"""vector[{len(self.lo_data)}] {self.lo_data_name};
                   vector[{len(self.hi_data)}] {self.hi_data_name};"""

    @trace
    def stan_data_card(self):
        """
        @returns Set one-sigma lower and upper values for additive corrections
        """
        return {self.lo_data_name: self.lo_data,
                self.hi_data_name: self.hi_data}

    @trace
    def stan_trans_pars(self):
        """
        @returns Linearly-interpolated additive correction to sample
        """
        return f"{self.sample.par_name} += interp_linear({self.par_name}, {self.lo_data_name}, {self.hi_data_name});"

    @trace
    def stan_model(self):
        """
        @returns Constrain additive correction by normal
        """
        return add_to_target("normal_lpdf", self.par_name, 0, 1)


class CorrelatedBinWiseFactor(CorrelatedBinWiseAdditive):
    """
    A bin-wise multiplicative modifier. The multiplicative factors are correlated
    """
    @trace
    def stan_trans_pars(self):
        """
        @returns Linearly-interpolated multiplicative correction to sample
        """
        return f"{self.sample.par_name} *= interp_linear({self.par_name}, {self.lo_data_name}, {self.hi_data_name});"


MODIFIERS = {
    "lumi": Factor,
    "normfactor": Factor,
    'shapesys': PoissonBinWiseFactor,
    'histosys': CorrelatedBinWiseAdditive,
    'normsys': CorrelatedBinWiseFactor,
    'staterror': NormalBinWiseFactor,
    "shapefactor": BinWiseFactor}


def find_modifier(modifier, *args, **kwargs):
    """
    @returns Modifier from hf modifier data
    """
    cls = MODIFIERS.get(modifier['type'])
    return cls(modifier, *args, **kwargs)


def order_modifiers(modifiers):
    """
    @returns Additive modifiers brought to front
    """
    add = [a for a in modifiers if isinstance(a, CorrelatedBinWiseAdditive)]
    other = [
        a for a in modifiers if not isinstance(
            a, CorrelatedBinWiseAdditive)]
    return add + other
