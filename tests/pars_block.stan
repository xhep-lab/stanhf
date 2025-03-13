parameters{
array[1 - fix_k_histosys] real<lower=lu_k_histosys.1, upper=lu_k_histosys.2> free_k_histosys;  
real<lower=lu_k_normsys.1, upper=lu_k_normsys.2> k_normsys;  
vector<lower=lu_k_shapesys.1, upper=lu_k_shapesys.2>[2] k_shapesys;  
vector<lower=lu_k_staterror.1, upper=lu_k_staterror.2>[2] k_staterror;  
real<lower=lu_lumi.1, upper=lu_lumi.2> lumi;  
real<lower=lu_k_normfactor.1, upper=lu_k_normfactor.2> k_normfactor;  
vector<lower=lu_k_shapefactor.1, upper=lu_k_shapefactor.2>[2] k_shapefactor;  
vector<lower=lu_l_shapesys.1, upper=lu_l_shapesys.2>[2] l_shapesys;  
vector<lower=lu_l_staterror.1, upper=lu_l_staterror.2>[2] l_staterror;  
vector<lower=lu_m_shapesys.1, upper=lu_m_shapesys.2>[2] m_shapesys;  
vector<lower=lu_m_staterror.1, upper=lu_m_staterror.2>[2] m_staterror;  
vector<lower=lu_n_shapesys.1, upper=lu_n_shapesys.2>[2] n_shapesys;  
vector<lower=lu_n_staterror.1, upper=lu_n_staterror.2>[2] n_staterror;  
}