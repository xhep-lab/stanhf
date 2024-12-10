transformed parameters{
vector[2] expected_singlechannel_signal = nominal_singlechannel_signal; // from Sample.stan_trans_pars [sample.py:L58]
expected_singlechannel_signal += term_interp(k_histosys, nominal_singlechannel_signal, lu_singlechannel_signal_histosys_k_histosys); // from HistoSys.stan_trans_pars [modifier.py:L274]
expected_singlechannel_signal *= factor_interp(k_normsys, lu_singlechannel_signal_normsys_k_normsys); // from NormSys.stan_trans_pars [modifier.py:L329]
expected_singlechannel_signal .*= k_shapesys; // from ShapeSys.stan_trans_pars [modifier.py:L185]
                   vector[2] expected_singlechannel_signal_shapesys_k_shapesys = k_shapesys .* observed_singlechannel_signal_shapesys_k_shapesys; // from ShapeSys.stan_trans_pars [modifier.py:L185]
vector[2] expected_singlechannel = expected_singlechannel_signal; // from Channel.stan_trans_pars [channel.py:L43]
}