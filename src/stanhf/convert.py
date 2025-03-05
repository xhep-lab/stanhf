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

import numpy as np
import pyhf
from cmdstanpy import format_stan_file, write_stan_json, compile_stan_file

from .channel import Channel
from .config import find_measureds, find_params, FreeParameter, FixedParameter, NullParameter
from .modifier import find_constraint, find_staterror, check_per_channel
from .stanstr import block, flatten, jlint, read_observed
from .par_names import get_stan_par_names, get_pyhf_par_data
from .tracer import mergetraced
from .run import perturb_param_file, run_pyhf_model, run_stanhf_model


VERSION = importlib.metadata.version(__package__)
CWD = os.path.dirname(os.path.realpath(__file__))
STAN_FUNCTIONS = os.path.join(CWD, "stanhf.stanfunctions")


def is_newer(a, b):
    """
    @returns Whether file a or code is newer than file b
    """
    if not os.path.isfile(b):
        return True
    return  max(os.path.getmtime(__file__), os.path.getmtime(a)) > os.path.getmtime(b)


class Convert:
    """
    Convert histfactory into Stan code
    """

    def __init__(self, hf_file_name, patch=None):
        """
        @param hf_file_name JSON file name
        """
        self.hf_file_name = hf_file_name
        self.patch = patch

    @cached_property
    def _workspace(self):
        """
        @returns Workspace, patched if necessary
        """
        with open(self.hf_file_name, encoding="utf-8") as hf_file:
            workspace = pyhf.Workspace(json.load(hf_file))

        if self.patch is None:
            return workspace

        patch_json_file_name, patch_number = self.patch

        with open(patch_json_file_name, encoding="utf-8") as patch_file:
            patch_set = pyhf.PatchSet(json.load(patch_file))

        patch = patch_set.patches[patch_number]
        return patch.apply(workspace)

    @cached_property
    def _root(self):
        """
        @returns Root for default file names
        """
        root = os.path.splitext(self.hf_file_name)[0]

        if self.patch is None:
            return root

        patch_json_file_name, patch_number = self.patch
        patch_root = os.path.splitext(
            os.path.split(patch_json_file_name)[1])[0]
        return f"{root}_{patch_root}_{patch_number}"

    def _metadata(self):
        """
        @returns Metadata for Stan program
        """
        hf_version = self._workspace.get("version")
        return f"""// histfactory json {self.hf_file_name}
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
        modifiers = flatten(c.modifiers for c in self._channels)
        check_per_channel(modifiers)
        return modifiers

    @cached_property
    def _non_null_modifiers(self):
        """
        @returns Non-null modifiers in hf
        """
        return [m for m in self._modifiers if not m.is_null]

    @cached_property
    def _filter_pars(self):
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
        return [[p.par_name for p in pars] for pars in self._filter_pars]

    @cached_property
    def par_size(self):
        """
        @returns Number of parameters, fixed parameters and null parameters
        """
        return [sum(max(p.par_size, 1) for p in pars) for pars in self._filter_pars]

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
        return (f"- pyhf file '{self.hf_file_name}'\n"
                f"{par} free parameters, {fixed} fixed parameters and {null} ignored null parameters\n"
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
        blocks = [self._metadata(),
                  self.functions_block(),
                  self.data_block(),
                  self.transformed_data_block(),
                  self.pars_block(),
                  self.transformed_pars_block(),
                  self.model_block(),
                  self.generated_quantities_block()]
        return "\n\n".join([b for b in blocks if b is not None])

    def write_stan_file(self, file_name=None):
        """
        Write Stan program to a file

        @returns File name of Stan program
        """
        if file_name is None:
            file_name = f"{self._root}.stan"

        if is_newer(self.hf_file_name, file_name):

            with open(file_name, "w", encoding="utf-8") as stan_file:
                stan_file.write(self.to_stan())

            try:
                format_stan_file(file_name, overwrite_file=True, backup=False)
            except (CalledProcessError, RuntimeError) as err:
                warnings.warn(f"did not lint --- {str(err)}")
        else:
            warnings.warn(
                f"not overwriting {file_name} as newer than {self.hf_file_name}")

        return file_name

    def write_stan_data_file(self, file_name=None):
        """
        Write Stan data to a file

        @returns File name of Stan data file
        """
        if file_name is None:
            file_name = f"{self._root}_data.json"

        if is_newer(self.hf_file_name, file_name):
            write_stan_json(file_name, self.data_card())
            jlint(file_name)
        else:
            warnings.warn(
                f"not overwriting {file_name} as newer than {self.hf_file_name}")

        return file_name

    def write_stan_init_file(self, file_name=None):
        """
        Write Stan initial values to a file

        @returns File name of Stan init file
        """
        if file_name is None:
            file_name = f"{self._root}_init.json"

        if is_newer(self.hf_file_name, file_name):
            write_stan_json(file_name, self.init_card())
            jlint(file_name)
        else:
            warnings.warn(
                f"not overwriting {file_name} as newer than {self.hf_file_name}")

        return file_name

    def write_to_disk(self):
        """
        Write Stan model, data and initial values to disk

        @returns File names of Stan model files
        """
        return self.write_stan_file(), self.write_stan_data_file(), self.write_stan_init_file()

    def build(self):
        """
        Build Stan model

        @returns File name of executable Stan model
        """
        stan_file_name = self.write_stan_file()
        return compile_stan_file(stan_file_name)

    def validate_target(self, rng=None):
        """
        Validates stanhf target against pyhf
        """
        data_file_name = self.write_stan_data_file()
        init_file_name = self.write_stan_init_file()
        exe_file_name = self.build()

        pars = perturb_param_file(init_file_name, rng)

        stanhf_target = run_stanhf_model(
            pars, data_file_name, exe_file_name)
        nhf_target = run_pyhf_model(pars, self._workspace)

        if not np.isclose(stanhf_target, nhf_target):
            raise RuntimeError(
                f"no agreement in target:\n"
                f"Stan = {stanhf_target}\n"
                f"pyhf = {nhf_target}\n"
                f"delta = {stanhf_target - nhf_target}\n"
                f"for pars = {pars}")

    def validate_par_names(self):
        """
        Validates stanhf parameter names and sizes against pyhf
        """
        pyhf_par_data = get_pyhf_par_data(self._workspace)
        stanhf_par_data = {m.par_name: max(1, m.par_size) for m in self._pars}

        if stanhf_par_data != pyhf_par_data:
            raise RuntimeError(
                "no agreement in parameter names & sizes:\n"
                f"Stanhf = {stanhf_par_data}\n"
                f"pyhf = {pyhf_par_data}")

        stanhf_par_names = self.par_names[0]
        stan_file_name = self.write_stan_file()
        stan_par_names = get_stan_par_names(stan_file_name)

        if set(stanhf_par_names) != set(stan_par_names):
            raise RuntimeError(
                "no agreement in parameter names:\n"
                f"Stanhf = {stanhf_par_names}\n"
                f"Stan = {stan_par_names}")
