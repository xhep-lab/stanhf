<h1 align="center">
 🌀 stanhf
</h1>

<div align="center">
<i>Convert a histfactory model into a Stan model. </i>
</div>
<br>

Convert a [histfactory](https://cds.cern.ch/record/1456844) declarative specification of a model in json into a [Stan](https://mc-stan.org/) model. Stan is a probabilistic programming language and a set of algorithms with automatic differentiation.

We follow the histfactory specification as closely as possible and the target function (log-likelihood) should match that from [pyhf](https://github.com/scikit-hep/pyhf) up to a constant term to within a negligible numerical difference. 

## Install

    pip install stanhf

At runtime, the first time you use stanhf it could install cmdstan if it isn't found. This is required to lint, validate and compile any Stan models, though stanhf can be used as a conversion tool without it.

## Run

Stanhf consists of one CLI. See

    stanhf --help

for details. Try e.g.,

    stanhf ./examples/model.json

This converts, compiles and validates the example model. The compiled model is a cmdstan executable. You can run the usual Stan algorithms (HMC, optimization etc) through this executable. 

## Workflows

See [EXAMPLE.md](EXAMPLE.md) for a walkthrough of how to run and analyse outpus from a compiled Stan model.
