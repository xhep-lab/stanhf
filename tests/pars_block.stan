parameters{
real<lower=lu_k_histosys.1, upper=lu_k_histosys.2> k_histosys; // from FreeParameter.stan_pars [config.py:L82]
real<lower=lu_k_normsys.1, upper=lu_k_normsys.2> k_normsys; // from FreeParameter.stan_pars [config.py:L82]
vector<lower=lu_k_shapesys.1, upper=lu_k_shapesys.2>[2] k_shapesys; // from FreeParameter.stan_pars [config.py:L82]
}