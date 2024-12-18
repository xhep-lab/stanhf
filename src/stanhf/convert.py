"""
Convert a histfactory json model to Stan!
=========================================
"""

import importlib.metadata
import json
import os
import warnings
from subprocess import CalledProcessError
from functools import cached_property

from cmdstanpy import format_stan_file, write_stan_json

from .channel import Channel
from .config import find_measureds, find_params, FreeParameter, FixedParameter, NullParameter
from .modifier import find_constraint, find_staterror
from .stanstr import block, flatten, jlint, read_observed, pyhf_par_names
from .tracer import mergetraced


VERSION = importlib.metadata.version(__package__)
CWD = os.path.dirname(os.path.realpath(__file__))
STAN_FUNCTIONS = os.path.join(CWD, "stanhf.stanfunctions")


class Convert:
    """
    Convert histfactory into Stan code
    """

    def __init__(self, hf_json_file_name):
        """
        @param hf_json_file_name JSON file name
        """
        self.hf_json_file_name = hf_json_file_name

        with open(hf_json_file_name, encoding="utf-8") as hf_file:
            self._workspace = json.load(hf_file)

    def metadata(self):
        """
        @returns Metadata for Stan program
        """
        hf_version = self._workspace.get("version")
        return f"""// histfactory json {self.hf_json_file_name}
                   // histfactory spec version {hf_version}
                   // converted with stanhf {VERSION}"""

    @cached_property
    def _samples(self):
        """
        @returns Samples for all channels
        """
        return flatten(c.samples for c in self._channels)

    @cached_property
    def _observed(self):
        """
        @returns Observed counts for each channel
        """
        return {k["name"]: read_observed(k["data"]) for k in self._workspace["observations"]}

    @cached_property
    def _channels(self):
        """
        @returns All channels
        """
        return [Channel(c, self._observed[c["name"]]) for c in self._workspace["channels"]]

    @cached_property
    def _config(self):
        """
        @returns Configuration block from hf program
        """
        try:
            pars = self._workspace["measurements"][0]["config"]["parameters"]
        except (KeyError, IndexError):
            warnings.warn("no configuration data found")
            return {}
        return {p["name"]: p for p in pars}

    @cached_property
    def _measureds(self):
        """
        @returns Measurements for Stan program
        """
        return find_measureds(self._config, self._modifiers)

    @cached_property
    def _pars(self):
        """
        @returns Parameters for Stan program
        """
        return find_params(self._config, self._modifiers)

    @cached_property
    def _constraints(self):
        """
        @returns Constraints for Stan program
        """
        return [find_constraint(self._modifiers)]

    @cached_property
    def _staterror(self):
        """
        @returns Combined statistical error constraints for Stan program
        """
        return flatten([find_staterror(c) for c in self._channels])

    @cached_property
    def _modifiers(self):
        """
        @returns Modifiers in hf
        """
        return flatten(c.modifiers for c in self._channels)

    @cached_property
    def _non_null_modifiers(self):
        """
        @returns Non-null modifiers in hf
        """
        return [m for m in self._modifiers if not m.is_null]

    @cached_property
    def pyhf_par_names(self):
        """
        @returns All parameter names in pyhf style
        """
        return pyhf_par_names({m.par_name: m.par_size for m in self._pars})

    @cached_property
    def filter_pars(self):
        """
        @returns Names of parameters, fixed parameters and null parameters
        """
        par = [p for p in self._pars if isinstance(p, FreeParameter)]
        fixed = [p for p in self._pars if isinstance(p, FixedParameter)]
        null = [p for p in self._pars if isinstance(p, NullParameter)]
        return par, fixed, null

    @cached_property
    def par_names(self):
        """
        @returns Names of parameters, fixed parameters and null parameters
        """
        return [[p.par_name for p in pars] for pars in self.filter_pars]

    @cached_property
    def par_size(self):
        """
        @returns Number of parameters, fixed parameters and null parameters
        """
        return [sum(max(p.par_size, 1) for p in pars) for pars in self.filter_pars]

    @cached_property
    def model_size(self):
        """
        @returns Number of channels, samples, and modifiers
        """
        channels = len(self._channels)
        samples = len(self._samples)
        non_null_modifiers = len(self._non_null_modifiers)
        null_modifiers = len(self._modifiers) - non_null_modifiers
        return channels, samples, non_null_modifiers, null_modifiers

    def __str__(self):
        """
        @returns Summary of model
        """
        par, fixed, null = self.par_size
        channels, samples, non_null_modifiers, null_modifiers = self.model_size
        return (f"- pyhf file '{self.hf_json_file_name}'\n"
                f"{par} free parameters, {fixed} fixed parameters and {null} parameters\n"
                f"{channels} channels with {samples} samples\n"
                f"{non_null_modifiers} modifiers and {null_modifiers} ignored null modifiers")

    @cached_property
    def _data(self):
        """
        @returns Representation of all elements in Stan program
        """
        return self._samples + self._measureds + self._non_null_modifiers + \
            self._pars + self._channels + self._constraints + self._staterror

    def functions_block(self):
        """
        @returns Functions block in Stan program
        """
        with open(STAN_FUNCTIONS, encoding="utf-8") as sf_file:
            functions = sf_file.read()
        return block("functions", [f"// [{STAN_FUNCTIONS}]", functions])

    def data_block(self):
        """
        @returns Data block in Stan program
        """
        return block("data", [e.stan_data() for e in self._data])

    def transformed_data_block(self):
        """
        @returns Transformed data block in Stan program
        """
        return block("transformed data", [
                     e.stan_trans_data() for e in self._data])

    def pars_block(self):
        """
        @returns Parameters block in Stan program
        """
        return block("parameters", [e.stan_pars() for e in self._data])

    def transformed_pars_block(self):
        """
        @returns Transformed parameters block in Stan program
        """
        return block("transformed parameters", [
                     e.stan_trans_pars() for e in self._data])

    def model_block(self):
        """
        @returns Model block in Stan program
        """
        return block("model", [e.stan_model() for e in self._data])

    def generated_quantities_block(self):
        """
        @returns Generated quantities block in Stan program
        """
        return block("generated quantities", [
                     e.stan_gen_quant() for e in self._data])

    def data_card(self):
        """
        @returns Data for Stan program
        """
        return mergetraced(e.stan_data_card() for e in self._data)

    def init_card(self):
        """
        @returns Initial parameter values for Stan program
        """
        return mergetraced(e.stan_init_card() for e in self._data)

    def to_stan(self):
        """
        @returns Blocks for Stan program
        """
        blocks = [self.metadata(),
                  self.functions_block(),
                  self.data_block(),
                  self.transformed_data_block(),
                  self.pars_block(),
                  self.transformed_pars_block(),
                  self.model_block(),
                  self.generated_quantities_block()]
        return "\n\n".join([b for b in blocks if b is not None])

    def write_stan_file(self, file_name, overwrite=True):
        """
        Write Stan program to a file
        """
        if overwrite or not os.path.isfile(file_name):
            with open(file_name, "w", encoding="utf-8") as stan_file:
                stan_file.write(self.to_stan())

            try:
                format_stan_file(file_name, overwrite_file=True, backup=False)
            except (CalledProcessError, RuntimeError) as err:
                warnings.warn(f"did not lint --- {str(err)}")

    def write_stan_data_file(self, file_name, overwrite=True):
        """
        Write Stan data to a file
        """
        if overwrite or not os.path.isfile(file_name):
            write_stan_json(file_name, self.data_card())
            jlint(file_name)

    def write_stan_init_file(self, file_name, overwrite=True):
        """
        Write Stan initial values to a file
        """
        if overwrite or not os.path.isfile(file_name):
            write_stan_json(file_name, self.init_card())
            jlint(file_name)


def convert(hf_json_file, overwrite=True):
    """
    @param hf_json_file Name of hf file
    @returns Root name of output files
    """
    root = os.path.splitext(hf_json_file)[0]

    convert_ = Convert(hf_json_file)
    convert_.write_stan_file(f"{root}.stan", overwrite)
    convert_.write_stan_data_file(f"{root}_data.json", overwrite)
    convert_.write_stan_init_file(f"{root}_init.json", overwrite)

    return root, convert_
