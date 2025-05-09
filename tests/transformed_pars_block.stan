transformed parameters{
vector[2] expected_singlechannel_signal = nominal_singlechannel_signal;  
vector[2] expected_singlechannel_background = nominal_singlechannel_background;  
vector[2] expected_secondchannel_signal = nominal_secondchannel_signal;  
vector[2] expected_secondchannel_background = nominal_secondchannel_background;  
real k_histosys = fix_k_histosys ? fixed_k_histosys : free_k_histosys[1];  
expected_singlechannel_signal += term_interp(k_histosys, nominal_singlechannel_signal, lu_singlechannel_signal_histosys_k_histosys);  
expected_singlechannel_signal *= factor_interp(k_normsys, lu_singlechannel_signal_normsys_k_normsys);  
expected_singlechannel_signal .*= k_shapesys;  
                   vector[2] expected_singlechannel_signal_shapesys_k_shapesys = k_shapesys .* observed_singlechannel_signal_shapesys_k_shapesys;  
expected_singlechannel_signal .*= k_staterror;  
expected_singlechannel_signal *= lumi;  
expected_singlechannel_signal *= k_normfactor;  
expected_singlechannel_signal .*= k_shapefactor;  
expected_singlechannel_background += term_interp(k_histosys, nominal_singlechannel_background, lu_singlechannel_background_histosys_k_histosys);  
expected_singlechannel_background *= factor_interp(k_normsys, lu_singlechannel_background_normsys_k_normsys);  
expected_singlechannel_background .*= l_shapesys;  
                   vector[2] expected_singlechannel_background_shapesys_l_shapesys = l_shapesys .* observed_singlechannel_background_shapesys_l_shapesys;  
expected_singlechannel_background .*= l_staterror;  
expected_singlechannel_background *= lumi;  
expected_singlechannel_background *= k_normfactor;  
expected_singlechannel_background .*= k_shapefactor;  
expected_secondchannel_signal += term_interp(k_histosys, nominal_secondchannel_signal, lu_secondchannel_signal_histosys_k_histosys);  
expected_secondchannel_signal *= factor_interp(k_normsys, lu_secondchannel_signal_normsys_k_normsys);  
expected_secondchannel_signal .*= m_shapesys;  
                   vector[2] expected_secondchannel_signal_shapesys_m_shapesys = m_shapesys .* observed_secondchannel_signal_shapesys_m_shapesys;  
expected_secondchannel_signal .*= m_staterror;  
expected_secondchannel_signal *= lumi;  
expected_secondchannel_signal *= k_normfactor;  
expected_secondchannel_signal .*= k_shapefactor;  
expected_secondchannel_background += term_interp(k_histosys, nominal_secondchannel_background, lu_secondchannel_background_histosys_k_histosys);  
expected_secondchannel_background *= factor_interp(k_normsys, lu_secondchannel_background_normsys_k_normsys);  
expected_secondchannel_background .*= n_shapesys;  
                   vector[2] expected_secondchannel_background_shapesys_n_shapesys = n_shapesys .* observed_secondchannel_background_shapesys_n_shapesys;  
expected_secondchannel_background .*= n_staterror;  
expected_secondchannel_background *= lumi;  
expected_secondchannel_background *= k_normfactor;  
expected_secondchannel_background .*= k_shapefactor;  
vector[2] expected_singlechannel = expected_singlechannel_signal + expected_singlechannel_background;  
vector[2] expected_secondchannel = expected_secondchannel_signal + expected_secondchannel_background;  
}