"""
Parse a channel from a hf model
===============================
"""

from .stanabc import Stan
from .sample import Sample
from .stanstr import join, add_to_target, flatten
from .metadata import add_metadata_comment, add_metadata_entry


class Channel(Stan):
    """
    Represent a single channel
    """

    def __init__(self, channel, observed):
        """
        @param channel hf channel
        @param observed Observed counts for channel
        """
        self.channel = channel
        self.observed = observed
        self.nbins = len(self.observed)
        self.name = channel["name"]
        self.expected_name = join("expected", self.name)
        self.observed_name = join("observed", self.name)

    @property
    def samples(self):
        """
        @returns Samples associated with this channel
        """
        return [Sample(s, self) for s in self.channel.get("samples", [])]

    @property
    def modifiers(self):
        """
        @returns Modifiers associated with this channel
        """
        return flatten([s.modifiers for s in self.samples])

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Total expected number of events from all samples in channel
        """
        total = " + ".join([s.par_name for s in self.samples])
        return f"vector[{self.nbins}] {self.expected_name} = {total};"

    @add_metadata_comment
    def stan_model(self):
        """
        @returns Poisson log-likelihood for total expected events in channel
        """
        return add_to_target(
            "poisson", self.observed_name, self.expected_name)

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare observed counts in channel
        """
        return f"array[{self.nbins}] int {self.observed_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Observed counts in channel
        """
        return {self.observed_name: self.observed}

    @add_metadata_comment
    def stan_gen_quant(self):
        """
        @returns Posterior predictive for counts in channel
        """
        rv_name = join("rv", self.expected_name)
        return f"array[{self.nbins}] int {rv_name} = poisson_rng({self.expected_name});"
