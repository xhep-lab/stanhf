<h1 align="center">
 🌀 stanhf
</h1>

<div align="center">
<i>Convert a histfactory model into a Stan model. </i>
</div>
<br>

Convert a [histfactory](https://cds.cern.ch/record/1456844) declarative specification of a model in json into a [Stan](https://mc-stan.org/) model. Stan is a probabilistic programming language and a set of algorithms with automatic differentiation.

We follow the histfactory specification as closely as possible and the target function (log-likelihood) should match that from [pyhf](https://github.com/scikit-hep/pyhf) to within a negligible numerical difference. 

## ✨ Install

    pipx install .

At runtime, the first time you use stanhf it could install cmdstan if it isn't found. This is required to lint, validate and compile any Stan models, though stanhf can be used as a conversion tool without it.

## Run

Stanhf consists of one CLI. See

    stanhf --help

for details. Try e.g.,

    stanhf ./examples/example.json

This converts, compiles and validates the example model. The compiled model is a cmdstan executable. You can run the usual Stan algorithms (HMC, optimization etc) through this executable. See [EXAMPLE.md](EXAMPLE.md) for a walkthrough.

## Differences with respect to pyhf

The only deliberate deviations from pyhf are the stricter interpretation of two edges cases:

- Applying the same type of modifiers to the same sample with the same name results in an error. In pyhf, these modifiers are overwritten
- Applying a staterror modifier with the same name in two different channels results in an error. In pyhf, these are silently treated as two different modifiers

However, stanhf might be more permissive with regard to other aspects of the hf specification. E.g.,

- stanhf does not require a poi to be specified
