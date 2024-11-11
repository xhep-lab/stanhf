"""
Parse a channel from a hf model
===============================
"""

from .stanabc import Stan
from .sample import Sample
from .stanstr import join, add_to_target
from .tracer import trace


class Channel(Stan):
    """
    Represent a single channel
    """

    def __init__(self, channel, data, suffix=False):
        """
        @param channel hf channel
        @param data Observed counts for channel
        @param suffix Whether to add suffix to Stan variable names
        """
        self.channel = channel
        self.data = [int(d) for d in data]
        self.suffix = suffix

        self.nbins = len(self.data)

        self.par_name = join("lambda", channel['name']) if suffix else "lambda"
        self.data_name = join("x", channel['name']) if suffix else "x"

    @property
    def samples(self):
        """
        @returns Samples associated with this channel
        """
        prefix = self.channel['name'] if self.suffix else None
        return [Sample(s, prefix) for s in self.channel["samples"]]

    @trace
    def stan_trans_pars(self):
        """
        @returns Total expected number of events from all samples in channel
        """
        total = " + ".join([s.par_name for s in self.samples])
        return f"vector<lower=0>[{self.nbins}] {self.par_name} = {total};"

    @trace
    def stan_model(self):
        """
        @returns Poisson log-likelihood for total expected events in channel
        """
        return add_to_target("poisson_lpmf", self.data_name, self.par_name)

    @trace
    def stan_data(self):
        """
        @returns Declare observed counts in channel
        """
        return f'array[{self.nbins}] int<lower=0> {self.data_name};'

    @trace
    def stan_data_card(self):
        """
        @returns Observed counts in channel
        """
        return {self.data_name: self.data}

    @trace
    def stan_gen_quant(self):
        """
        @returns Posterior predictive for counts in channel
        """
        return f"array[{self.nbins}] int {join('rv', self.par_name)} = poisson_rng({self.par_name});"
