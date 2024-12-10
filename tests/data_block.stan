data{
vector[2] nominal_singlechannel_signal; // from Sample.stan_data [sample.py:L65]
tuple(vector[2], vector[2]) lu_singlechannel_signal_histosys_k_histosys; // from HistoSys.stan_data [modifier.py:L258]
tuple(real, real) lu_singlechannel_signal_normsys_k_normsys; // from NormSys.stan_data [modifier.py:L313]
vector[2] rel_error_singlechannel_signal_shapesys_k_shapesys; // from ShapeSys.stan_data [modifier.py:L208]
tuple(real, real) lu_k_histosys; // from FreeParameter.stan_data [config.py:L99]
tuple(real, real) lu_k_normsys; // from FreeParameter.stan_data [config.py:L99]
tuple(vector[2], vector[2]) lu_k_shapesys; // from FreeParameter.stan_data [config.py:L99]
array[2] int observed_singlechannel; // from Channel.stan_data [channel.py:L59]
}