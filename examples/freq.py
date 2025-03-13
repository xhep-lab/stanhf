"""
Compare pyhf and Stan model for frequentist statistics
======================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt

import pyhf
from pyhf.contrib.viz import brazil

from stanhf.freq import MockModel, set_pyhf_stan


with open("example.json", encoding="utf-8") as f:
    workspace = pyhf.Workspace(json.load(f))

model = workspace.model()
data = workspace.data(model)


obs_limit, exp_limits, (poi_tests, tests) = pyhf.infer.intervals.upper_limits.upper_limit(
    data, model, np.linspace(0, 5, 1), level=0.05, return_results=True)

print(f'expected upper limits: {exp_limits}')
print(f'observed upper limit : {obs_limit}')

fig, ax = plt.subplots(figsize=(10, 7))
artists = brazil.plot_results(poi_tests, tests, test_size=0.05, ax=ax)
plt.savefig("native.pdf")

model = MockModel("example.stan", "example_data.json", "example_init.json")

with set_pyhf_stan():

    obs_limit, exp_limits, (poi_tests, tests) = pyhf.infer.intervals.upper_limits.upper_limit(
        model.data(), model, np.linspace(0, 5, 100), level=0.05, return_results=True
    )

print(f'expected upper limits: {exp_limits}')
print(f'observed upper limit : {obs_limit}')

fig, ax = plt.subplots(figsize=(10, 7))
artists = brazil.plot_results(poi_tests, tests, test_size=0.05, ax=ax)
plt.savefig("stan.pdf")
