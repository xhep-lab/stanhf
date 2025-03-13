data{
vector[2] nominal_singlechannel_signal;  
vector[2] nominal_singlechannel_background;  
vector[2] nominal_secondchannel_signal;  
vector[2] nominal_secondchannel_background;  
  
                int<lower=0, upper=1> fix_k_histosys;  
                real fixed_k_histosys;  
                tuple(real, real) lu_k_histosys;  
                  
tuple(real, real) lu_k_normsys;  
tuple(vector[2], vector[2]) lu_k_shapesys;  
tuple(vector[2], vector[2]) lu_k_staterror;  
tuple(real, real) lu_lumi;  
tuple(real, real) lu_k_normfactor;  
tuple(vector[2], vector[2]) lu_k_shapefactor;  
tuple(vector[2], vector[2]) lu_l_shapesys;  
tuple(vector[2], vector[2]) lu_l_staterror;  
tuple(vector[2], vector[2]) lu_m_shapesys;  
tuple(vector[2], vector[2]) lu_m_staterror;  
tuple(vector[2], vector[2]) lu_n_shapesys;  
tuple(vector[2], vector[2]) lu_n_staterror;  
tuple(real, real) normal_lumi;  
tuple(vector[2], vector[2]) lu_singlechannel_signal_histosys_k_histosys;  
tuple(real, real) lu_singlechannel_signal_normsys_k_normsys;  
vector[2] rel_error_singlechannel_signal_shapesys_k_shapesys;  
vector[2] stdev_singlechannel_signal_staterror_k_staterror;  
tuple(vector[2], vector[2]) lu_singlechannel_background_histosys_k_histosys;  
tuple(real, real) lu_singlechannel_background_normsys_k_normsys;  
vector[2] rel_error_singlechannel_background_shapesys_l_shapesys;  
vector[2] stdev_singlechannel_background_staterror_l_staterror;  
tuple(vector[2], vector[2]) lu_secondchannel_signal_histosys_k_histosys;  
tuple(real, real) lu_secondchannel_signal_normsys_k_normsys;  
vector[2] rel_error_secondchannel_signal_shapesys_m_shapesys;  
vector[2] stdev_secondchannel_signal_staterror_m_staterror;  
tuple(vector[2], vector[2]) lu_secondchannel_background_histosys_k_histosys;  
tuple(real, real) lu_secondchannel_background_normsys_k_normsys;  
vector[2] rel_error_secondchannel_background_shapesys_n_shapesys;  
vector[2] stdev_secondchannel_background_staterror_n_staterror;  
array[2] int observed_singlechannel;  
array[2] int observed_secondchannel;  
}