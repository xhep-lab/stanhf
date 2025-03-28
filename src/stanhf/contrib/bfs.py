"""
Bayes factor surface
====================
"""

import matplotlib.pyplot as plt
import numpy as np

from arviz.stats.density_utils import _kde_linear
from arviz.data.utils import extract


def compute_bfs(idata, var_name, prior=None, ref_val=0.):
    """
    :param idata: arviz data
    :param var_name: Variance name for Bayes factor surface
    :param prior: Prior
    :param ref_val: Reference value to compute Bayes factor surface wrt

    :returns: grid and Bayes factor surface values at grid
    """
    posterior = extract(idata, var_names=var_name).values
    posterior_grid, posterior_pdf = _kde_linear(posterior)
    posterior_at_ref_val = np.interp(ref_val, posterior_grid, posterior_pdf)
    posterior_ratio = posterior_pdf / posterior_at_ref_val

    if prior is not None:
        prior_grid, prior_pdf = _kde_linear(prior)
        prior_at_ref_val = np.interp(ref_val, prior_grid, prior_pdf)
        prior_at_posterior_grid = np.interp(posterior_grid, prior_grid, prior_pdf)
        prior_ratio = prior_at_posterior_grid / prior_at_ref_val
    else:
        prior_ratio = 1.

    bfs = posterior_ratio / prior_ratio

    return posterior_grid, bfs


def plot_bfs(idata, var_name, prior=None, ref_val=0., ax=None, show=False, mark=None):
    """
    Makes a plot of Bayes factor surface
    """
    if ax is None:
        ax = plt.gca()

    if mark is None:
        mark = [0.05, 0.01]

    grid, bfs = compute_bfs(idata, var_name, prior, ref_val)
    par = [grid[np.argmin(np.abs(bfs - m))] for m in mark]

    ax.plot(grid, bfs)
    ax.set_xlabel(var_name)
    ax.set_ylabel(f"BFS versus {var_name} = {ref_val}")

    trans = ax.get_xaxis_transform()

    for p, m in zip(par, mark):
        plt.axvline(p, linestyle=':', color='C1')
        plt.text(p, 0.5, f'BFS = {m}', rotation=270,
                 transform=trans, color='C1', va="bottom")

    if show:
        plt.show()

    return grid, bfs, ax
