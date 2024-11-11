"""
Parse a sample
==============
"""

from .stanabc import Stan
from .stanstr import join
from .modifier import find_modifier
from .tracer import trace


class Sample(Stan):
    """
    Sample e.g., signal or background, contribution to a channel
    """

    def __init__(self, sample, prefix=None):
        """
        @param sample hf sample
        @param prefix Whether to prefix variable names by channel
        """
        self.sample = sample

        self.data = sample["data"]
        self.nbins = len(self.data)

        self.par_name = join(
            prefix, sample["name"]) if prefix else sample["name"]
        self.data_name = join("x", self.par_name)

    @property
    def modifiers(self):
        """
        @returns Modifiers associated with this sample
        """
        return [find_modifier(m, self)
                for m in self.sample["modifiers"]]

    @trace
    def stan_trans_pars(self):
        """
        @returns Declare and set expected events in this channel

        This can be later modified by the modifiers.
        """
        return f"vector<lower=0>[{self.nbins}] {self.par_name} = {self.data_name};"

    @trace
    def stan_data(self):
        """
        @returns Declare data for this sample (set in data card)
        """
        return f'vector<lower=0>[{self.nbins}] {self.data_name};'

    @trace
    def stan_data_card(self):
        """
        @returns Data for this sample
        """
        return {self.data_name: self.data}
