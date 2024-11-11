"""
Convert a histfactory json model to Stan!
=========================================
"""

import os
import json
import warnings

from subprocess import CalledProcessError
from cmdstanpy import format_stan_file, write_stan_json

from .channel import Channel
from .config import find_measureds, find_params
from .modifier import order_modifiers
from .stanstr import block, merge, flatten
from . import __version__


CWD = os.path.dirname(os.path.realpath(__file__))
STAN_FUNCTIONS = os.path.normpath(os.path.join(
    CWD, "..", "stanfunctions", "stanhf.stanfunctions"))


class Convert:
    """
    Convert a histfactory into Stan
    """

    def __init__(self, hf_json_file_name):
        self.hf_json_file_name = hf_json_file_name

        with open(hf_json_file_name, encoding="utf-8") as hff:
            self.workspace = json.load(hff)

    def metadata(self):
        hf_version = self.workspace.get("version")
        return (f"// histfactory json {self.hf_json_file_name}\n"
                f"// histfactory spec version {hf_version}\n"
                f"// converted with stanhf {__version__}")

    @property
    def samples(self):
        return flatten(c.samples for c in self.channels)

    @property
    def channels(self):
        observed = {k['name']: k['data']
                    for k in self.workspace["observations"]}
        suffix = len(self.workspace["channels"]) != 1
        return [Channel(c, observed[c['name']], suffix)
                for c in self.workspace["channels"]]

    @property
    def _config(self):
        list_ = self.workspace["measurements"][0]["config"]["parameters"]
        return {o['name']: o for o in list_}

    @property
    def measureds(self):
        return find_measureds(self._config)

    @property
    def pars(self):
        return find_params(self._config, self.modifiers)

    @property
    def modifiers(self):
        return order_modifiers(
            flatten(m for s in self.samples for m in s.modifiers))

    @property
    def data(self):
        return self.samples + self.channels + self.measureds + self.modifiers + self.pars

    def functions_block(self):
        with open(STAN_FUNCTIONS, encoding="utf-8") as sff:
            functions = sff.read()
        return block("functions", [f"// [{STAN_FUNCTIONS}]", functions])

    def data_block(self):
        return block("data", [e.stan_data() for e in self.data])

    def transformed_data_block(self):
        return block("transformed data", [
                     e.stan_trans_data() for e in self.data])

    def pars_block(self):
        return block("parameters", [e.stan_pars() for e in self.data])

    def transformed_pars_block(self):
        return block("transformed parameters", [
                     e.stan_trans_pars() for e in self.data])

    def model_block(self):
        return block("model", [e.stan_model() for e in self.data])

    def generated_quantities_block(self):
        return block("generated quantities", [
                     e.stan_gen_quant() for e in self.data])

    def data_card(self):
        return merge(e.stan_data_card() for e in self.data)

    def init_card(self):
        return merge(e.stan_init_card() for e in self.data)

    def to_stan(self):
        return "\n\n".join([self.metadata(),
                            self.functions_block(),
                            self.data_block(),
                            self.transformed_data_block(),
                            self.pars_block(),
                            self.transformed_pars_block(),
                            self.model_block(),
                            self.generated_quantities_block()])

    def write_stan_file(self, file_name, overwrite=True):

        if overwrite or not os.path.isfile(file_name):
            with open(file_name, "w", encoding="utf-8") as stf:
                stf.write(self.to_stan())

            try:
                format_stan_file(file_name, overwrite_file=True, backup=False)
            except CalledProcessError as cpe:
                warnings.warn(f"did not lint --- {str(cpe)}")

    def write_stan_data_file(self, file_name, overwrite=True):
        if overwrite or not os.path.isfile(file_name):
            write_stan_json(file_name, self.data_card())

    def write_stan_init_file(self, file_name, overwrite=True):
        if overwrite or not os.path.isfile(file_name):
            write_stan_json(file_name, self.init_card())


def converter(hf_json_file, overwrite=True):
    """
    @param hf_json_file Name of hf file
    @returns Root name of output files
    """
    root = os.path.splitext(hf_json_file)[0]

    convert = Convert(hf_json_file)
    convert.write_stan_file(f"{root}.stan", overwrite)
    convert.write_stan_data_file(f"{root}_data.json", overwrite)
    convert.write_stan_init_file(f"{root}_init.json", overwrite)

    return root
