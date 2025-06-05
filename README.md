<h1 align="center">
 🌀 stanhf
</h1>

<h3 align="center">
<i>Convert a histfactory model into a Stan model</i>
</h3>

<div align="center">
 
[![arXiv](https://img.shields.io/badge/arXiv-12503.22188-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2503.22188)
[![GitHub License](https://img.shields.io/github/license/xhep-lab/stanhf?style=for-the-badge)](https://github.com/xhep-lab/stanhf?tab=GPL-3.0-1-ov-file#)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/xhep-lab/stanhf/python-app.yml?style=for-the-badge)](https://github.com/xhep-lab/stanhf/actions)
![GitHub Tag](https://img.shields.io/github/v/tag/xhep-lab/stanhf?style=for-the-badge)
![PyPI - Version](https://img.shields.io/pypi/v/stanhf?style=for-the-badge)
</div>


<br>

Convert a [histfactory](https://cds.cern.ch/record/1456844) declarative specification of a model in json into a [Stan](https://mc-stan.org/) model. Stan is a probabilistic programming language and a set of algorithms with automatic differentiation.

We follow the histfactory specification as closely as possible and the target function (log-likelihood) should match that from [pyhf](https://github.com/scikit-hep/pyhf) up to a constant term to within a negligible numerical difference. 

See the paper at [[2503.22188](https://arxiv.org/abs/2503.22188)] for more information.

## Install

    pip install stanhf

At runtime, the first time you use stanhf it could install cmdstan if it isn't found. This is required to lint, validate and compile any Stan models, though stanhf can be used as a conversion tool without it.

## Run

Stanhf consists of one CLI. See

    stanhf --help

for details. Try e.g.,

    stanhf ./examples/normfactor.json

This converts, compiles and validates the example model. The compiled model is a cmdstan executable. You can run the usual Stan algorithms (HMC, optimization etc) through this executable. 

## Workflows

See [EXAMPLE.md](EXAMPLE.md) for a walkthrough of how to run and analyse outpus from a compiled Stan model.
