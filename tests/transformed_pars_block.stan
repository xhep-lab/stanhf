transformed parameters{
vector[2] expected_singlechannel_signal = nominal_singlechannel_signal; // from Sample.stan_trans_pars [sample.py:L55]
expected_singlechannel_signal += term_interp(k_histosys, nominal_singlechannel_signal, lu_singlechannel_signal_histosys_k_histosys); // from HistoSys.stan_trans_pars [modifier.py:L234]
vector[2] expected_singlechannel = expected_singlechannel_signal; // from Channel.stan_trans_pars [channel.py:L43]
}