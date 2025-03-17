"""
Frequentist inference using pyhf
================================
"""

import json

import numpy as np

import pyhf
import cmdstanpy


class StanOptimizer:
    """
    Optimize Stan model using Stan's built in optimizer
    """

    name = "Stan optimizer"

    @staticmethod
    def minimize(_objective,
                 data,
                 pdf,
                 _inits,
                 _bounds,
                 fixed_vals=None,
                 return_fitted_val=False,
                 return_result_obj=False,
                 **_kwargs):

        assert _objective is None or _objective is pyhf.infer.mle.twice_nll
        assert _inits is None or _inits == pdf.config.suggested_init(
        ) or _inits[0] == fixed_vals[0][1]
        assert _bounds is None or _bounds == pdf.config.suggested_bounds()

        poi_name = pdf.poi_name()
        inits = pdf.inits()
        data = data.copy()

        if fixed_vals:
            data[f"fix_{poi_name}"] = 1
            data[f"fixed_{poi_name}"] = fixed_vals[0][1]
            inits[f"free_{poi_name}"] = []

        result = pdf.model.optimize(data=data, inits=inits)

        two_nll = -2. * result.optimized_params_dict["lp__"]
        params = np.array(result.optimized_params_np)

        if poi_name:
            params[pdf.config.poi_index] = result.optimized_params_dict[poi_name]

        _returns = [params]

        if return_fitted_val:
            _returns.append(two_nll)

        if return_result_obj:
            _returns.append(result)

        return tuple(_returns) if len(_returns) > 1 else _returns[0]


class MockConfig:
    """
    Mock of pyhf config object. The contents are not used
    """

    poi_index = 0

    def suggested_init(self):
        return [0]

    def suggested_bounds(self):
        return [[0., np.inf]]

    def suggested_fixed(self):
        return [0]


class MockModel:
    """
    Mock of pyhf model
    """

    def __init__(self, stan_file, data_file, init_file):
        self.data_file = data_file
        self.stan_file = stan_file
        self.init_file = init_file
        self.config = MockConfig()
        self.model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    def data(self):
        with open(self.data_file, encoding="utf-8") as f:
            return json.load(f)

    def inits(self):
        with open(self.init_file, encoding="utf-8") as f:
            return json.load(f)

    def expected_data(self, params):
        """
        Generate expected data for a given set of parameters
        """

        fixed_vals = [(self.config.poi_index, params[self.config.poi_index])]
        result = StanOptimizer.minimize(
            None,
            self.data(),
            self,
            None,
            None,
            fixed_vals=fixed_vals,
            return_result_obj=True)[1]

        params = result.stan_variables()
        data = self.data()

        for k in params:
            if k.startswith("expected_"):
                n = k.replace("expected", "observed")
                data[n] = np.floor(params[k]).astype(int)

        return data

    def poi_name(self):
        """
        @returns Name of POI from Stan model
        """
        for k in self.data():
            if k.startswith("fix_"):
                return k[len("fix_"):]
        return None


class set_pyhf_stan:
    """
    Sets pyhf backend to be Stan; this works only for the mocked models
    """

    def __init__(self):
        self.org = pyhf.get_backend()

    def __enter__(self):
        pyhf.set_backend("numpy", custom_optimizer=StanOptimizer)

    def __exit__(self, *args, **kwargs):
        pyhf.set_backend(*self.org)
