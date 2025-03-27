"""
Frequentist inference using pyhf
================================

pyhf sees a one-dimensional version of the full model, with the POI in index 0
"""

import json

import numpy as np

import pyhf
import cmdstanpy


def run(pdf, data, bounds, inits, fixed_vals):
    """
    @returns Stan optimize result
    """
    _inits = pdf.config.inits.copy()
    _data = data.copy()
    bounds = bounds or pdf.config.suggested_bounds()
    inits = inits or pdf.config.suggested_init()
    poi_name = pdf.config.poi_name

    if fixed_vals:
        _data[f"fix_{poi_name}"] = 1
        _data[f"fixed_{poi_name}"] = fixed_vals[0][1]
        _inits[f"free_{poi_name}"] = []
    else:
        _data[f"lu_{poi_name}"] = (bounds[0][0], bounds[0][1])
        _inits[f"free_{poi_name}"] = inits

    return pdf.model.optimize(data=_data, inits=_inits)


class StanOptimizer:
    """
    Optimize Stan model using Stan's built in optimizer
    """

    name = "Stan optimizer"

    @staticmethod
    def minimize(_objective,
                 data,
                 pdf,
                 inits,
                 bounds,
                 fixed_vals=None,
                 return_fitted_val=False,
                 return_result_obj=False,
                 **_kwargs):
        """
        Signature matches pyhf optimizers. Works in tandem with the MockPyhfModel
        """
        assert _objective is None or _objective is pyhf.infer.mle.twice_nll
        assert isinstance(pdf, MockPyhfModel)

        result = run(pdf, data, bounds, inits, fixed_vals)

        two_nll = -2. * result.optimized_params_dict["lp__"]
        params = np.array(result.optimized_params_np)

        # swap elements so that poi in expected place

        poi_fit = result.optimized_params_dict[pdf.config.poi_name]
        poi_index = np.where(params == poi_fit)[0][0]
        params[[poi_index, pdf.config.poi_index]
               ] = params[[pdf.config.poi_index, poi_index]]

        _returns = [params]

        if return_fitted_val:
            _returns.append(two_nll)

        if return_result_obj:
            _returns.append(result)

        return tuple(_returns) if len(_returns) > 1 else _returns[0]


class MockPyhfConfig:
    """
    Mock of pyhf config object

    Reads bounds, initial values, POI etc from the Stan json files
    """

    poi_index = 0
    npars = 0

    def __init__(self, data_file, init_file):
        self.data_file = data_file
        self.init_file = init_file

    @property
    def inits(self):
        with open(self.init_file, encoding="utf-8") as f:
            return json.load(f)

    @property
    def data(self):
        with open(self.data_file, encoding="utf-8") as f:
            return json.load(f)

    def suggested_init(self):
        return self.inits[f"free_{self.poi_name}"]

    def suggested_bounds(self):
        lu = self.data.get(f"lu_{self.poi_name}")
        return [[lu["1"], lu["2"]]]

    def suggested_fixed(self):
        return [False]

    def par_slice(self, _):
        return slice(0, 1, 1)

    @property
    def poi_name(self):
        """
        @returns Name of POI from Stan model
        """
        for k in self.data:
            if k.startswith("fix_"):
                return k[len("fix_"):]

        raise RuntimeError("Could not find a POI; did you declare one?")


class MockPyhfModel:
    """
    Mock of pyhf model that works in tandem with the StanOptimize optimizer
    """

    def __init__(self, stan_file, data_file, init_file):
        self.config = MockPyhfConfig(data_file, init_file)
        self.model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    def expected_data(self, params):
        """
        Generate expected data for a given set of parameters
        """
        fixed_vals = [(self.config.poi_index, params[self.config.poi_index])]
        data = self.config.data.copy()
        result = StanOptimizer.minimize(
            None,
            data,
            self,
            None,
            None,
            fixed_vals=fixed_vals,
            return_result_obj=True)[1]

        params = result.stan_variables()

        for k in params:
            if k.startswith("expected_"):
                n = k.replace("expected", "observed")
                data[n] = np.floor(params[k]).astype(int)

        return data


class mock_pyhf_backend:
    """
    Sets pyhf backend to be Stan; this works only for the mocked models
    """

    def __init__(self):
        self.org = pyhf.get_backend()

    def __enter__(self):
        pyhf.set_backend("numpy", custom_optimizer=StanOptimizer)

    def __exit__(self, *args, **kwargs):
        pyhf.set_backend(*self.org)
