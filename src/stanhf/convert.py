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
from .config import find_measureds, find_params, FreeParameter, FixedParameter, NullParameter, POI
from .modifier import find_constraints, find_staterror, check_per_channel
from .stanstr import block, flatten, format_json_file, read_observed, remove_prefix
from .pars import get_stan_par_names, get_pyhf_par_data
from .metadata import merge_metadata
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
    return max(os.path.getmtime(__file__), os.path.getmtime(a)) > os.path.getmtime(b)


class Convert:
    """
    Convert histfactory into Stan code
    """

    def __init__(self, hf_file_name, patch=None):
        """
        @param hf_file_name JSON file name
        @param patch file name and number of a patchset
        """
        self.hf_file_name = hf_file_name
        self.patch = patch

    @cached_property
    def _patch(self):
        """
        @returns Patch with added metadata, if present
        """
        if self.patch is None:
            return None

        patch_file_name, patch_number = self.patch

        with open(patch_file_name, encoding="utf-8") as patch_file:
            patch_set = pyhf.PatchSet(json.load(patch_file))

        patch = patch_set.patches[patch_number]

        patch._metadata = patch.metadata | patch_set.metadata
        patch._metadata["version"] = patch_set.version

        return patch

    @cached_property
    def _workspace(self):
        """
        @returns Workspace, patched if necessary
        """
        with open(self.hf_file_name, encoding="utf-8") as hf_file:
            try:
                hf = json.load(hf_file)
            except json.decoder.JSONDecodeError as e:
                raise IOError(
                    f"could not read {self.hf_file_name} - is it a valid json file?") from e

        workspace = pyhf.Workspace(hf)

        if self._patch is None:
            return workspace

        return self._patch.apply(workspace)

    @cached_property
    def _root(self):
        """
        @returns Root for default file names
        """
        root = os.path.splitext(self.hf_file_name)[0]

        if self._patch is None:
            return root

        return f"{root}_{self._patch.name}"

    def _stanhf_metadata(self):
        """
        @returns Metadata for Stan program
        """
        hf_version = self._workspace.get("version")

        return f"""// histfactory json {self.hf_file_name}
                   // histfactory spec version {hf_version}
                   // converted with stanhf {VERSION}"""

    def _patch_metadata(self):
        """
        @returns Metadata for pyhf patch
        """
        if self.patch is None:
            return "// no patch applied"

        return f"""
                // description: {self._patch.metadata['description']}
                // patchset id: {self._patch.metadata['analysis_id']}
                // version: {self._patch.metadata['version']}
                // patch: {self._patch.name}
                """

    def _metadata(self):
        """
        @returns Combined metadata for program
        """
        return self._stanhf_metadata() + "\n\n" + self._patch_metadata()

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
    def _poi(self):
        """
        @returns POI
        """
        try:
            return self._workspace["measurements"][0]["config"]["poi"]
        except (KeyError, IndexError):
            warnings.warn("no configuration data found")
            return None

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
        return find_params(self._poi, self._config, self._modifiers)

    @cached_property
    def _constraints(self):
        """
        @returns Constraints for Stan program
        """
        return find_constraints(self._modifiers)

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
        par = [p for p in self._pars if isinstance(p, (POI, FreeParameter))]
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
        patch = f"'{self._patch.name}'" if self._patch else "no"

        return (f"hf file '{self.hf_file_name}' with {patch} patch applied:\n"
                f"- {par} free parameters, {fixed} fixed parameters and {null} ignored null parameters\n"
                f"- {channels} channels with {samples} samples\n"
                f"- {non_null_modifiers} modifiers and {null_modifiers} ignored null modifiers")

    @cached_property
    def _data(self):
        """
        @returns Representation of all elements in Stan program
        """
        return self._samples + self._pars + self._measureds + self._non_null_modifiers + \
            self._channels + self._constraints + self._staterror

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
        return merge_metadata([e.stan_data_card() for e in self._data])

    def init_card(self):
        """
        @returns Initial parameter values for Stan program
        """
        return merge_metadata([e.stan_init_card() for e in self._data])

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
            format_json_file(file_name)
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
            format_json_file(file_name)
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

    def build(self, stan_file_name=None):
        """
        Build Stan model

        @returns File name of executable Stan model
        """
        if stan_file_name is None:
            stan_file_name = self.write_stan_file()
        return compile_stan_file(stan_file_name)

    def validate_target(self, exe_file_name=None, stan_file_name=None, data_file_name=None, init_file_name=None, rng=None):
        """
        Validates stanhf target against pyhf
        """
        if data_file_name is None:
            data_file_name = self.write_stan_data_file()

        if init_file_name is None:
            init_file_name = self.write_stan_init_file()

        if exe_file_name is None:
            exe_file_name = self.build(stan_file_name)

        a = perturb_param_file(init_file_name, rng)
        b = perturb_param_file(init_file_name, rng)

        stanhf_delta = run_stanhf_model(
            b, data_file_name, exe_file_name) - run_stanhf_model(a, data_file_name, exe_file_name)
        nhf_delta = run_pyhf_model(
            b, self._workspace) - run_pyhf_model(a, self._workspace)

        if not np.isclose(stanhf_delta, nhf_delta):
            raise RuntimeError(
                f"no agreement in delta log-like:\n"
                f"Stan = {stanhf_delta}\n"
                f"pyhf = {nhf_delta}\n"
                f"difference = {stanhf_delta - nhf_delta}\n"
                f"for a = {a} and b = {b}")

    def validate_par_names(self, stan_file_name=None):
        """
        Validates stanhf parameter names and sizes against pyhf
        """
        if stan_file_name is None:
            stan_file_name = self.write_stan_file()

        pyhf_par_data = get_pyhf_par_data(self._workspace)
        stanhf_par_data = {remove_prefix(m.par_name, "free_"): max(
            1, m.par_size) for m in self._pars}

        if stanhf_par_data != pyhf_par_data:
            raise RuntimeError(
                "no agreement in parameter names & sizes:\n"
                f"Stanhf = {stanhf_par_data}\n"
                f"pyhf = {pyhf_par_data}\n"
                f"difference = {set(stanhf_par_data) ^ set(pyhf_par_data)}")

        stanhf_par_names = sorted(self.par_names[0])
        stan_par_names = sorted([remove_prefix(p, "free_")
                                for p in get_stan_par_names(stan_file_name)])

        if set(stanhf_par_names) != set(stan_par_names):
            raise RuntimeError(
                "no agreement in parameter names:\n"
                f"Stanhf = {stanhf_par_names}\n"
                f"Stan = {stan_par_names}",
                f"difference = {stanhf_par_names ^ stan_par_names}")
