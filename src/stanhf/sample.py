"""
Parse a sample
==============
"""

import warnings

from functools import cached_property

from .stanabc import Stan
from .stanstr import join
from .modifier import find_modifier, order_modifiers
from .metadata import add_metadata_comment, add_metadata_entry


class Sample(Stan):
    """
    Sample e.g., signal or background, contribution to a channel
    """

    def __init__(self, sample, channel):
        """
        @param sample hf sample
        @param channel Channel of this sample
        """
        self.sample = sample
        self.channel = channel

        self.nominal = sample["data"]
        self.nbins = channel.nbins

        self.name = join(channel.name, sample["name"])
        self.par_name = join("expected", self.name)
        self.nominal_name = join("nominal", self.name)

    @cached_property
    def modifiers(self):
        """
        @returns Ordered modifiers associated with this sample

        Ordered such that additive are first and repeated type/name modifiers are
        overwritten
        """
        modifiers = [
            find_modifier(
                m, self) for m in self.sample.get(
                "modifiers", [])]

        names = [m.name for m in modifiers]
        repeated = set(n for n in names if names.count(n) > 1)

        if repeated:
            warnings.warn(
                f"repeated type/name modifiers are overwritten: {repeated}")

        modifiers = {m.name: m for m in modifiers}.values()
        return order_modifiers(modifiers)

    @add_metadata_comment
    def stan_trans_pars(self):
        """
        @returns Declare and set expected events in this channel
        """
        return f"vector[{self.nbins}] {self.par_name} = {self.nominal_name};"

    @add_metadata_comment
    def stan_data(self):
        """
        @returns Declare data for this sample
        """
        return f"vector[{self.nbins}] {self.nominal_name};"

    @add_metadata_entry
    def stan_data_card(self):
        """
        @returns Set data for this sample
        """
        return {self.nominal_name: self.nominal}
