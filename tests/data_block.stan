data{
vector[2] nominal_singlechannel_signal; // from Sample.stan_data [sample.py:L62]
tuple(vector[2], vector[2]) lu_singlechannel_signal_histosys_k_histosys; // from HistoSys.stan_data [modifier.py:L220]
tuple(real, real) lu_k_histosys; // from Parameter.stan_data [config.py:L84]
array[2] int observed_singlechannel; // from Channel.stan_data [channel.py:L59]
}