"""
Ease conversion of histfactory json to Stan
===========================================
"""

from abc import ABC


class Stan(ABC):
    """
    Represent blocks of a Stan model
    """

    def stan_data(self):
        """
        @returns Stan data block
        """

    def stan_trans_data(self):
        """
        @returns Stan transformed data block
        """

    def stan_pars(self):
        """
        @returns Stan parameters block
        """

    def stan_trans_pars(self):
        """
        @returns Stan transformed parameters block
        """

    def stan_model(self):
        """
        @returns Stan model block
        """

    def stan_gen_quant(self):
        """
        @returns Stan generated quantities block
        """

    def stan_data_card(self):
        """
        @returns Stan data card
        """

    def stan_init_card(self):
        """
        @returns Stan initialization parameters
        """
