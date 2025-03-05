transformed data{
vector[2] observed_singlechannel_signal_shapesys_k_shapesys = square(nominal_singlechannel_signal ./ rel_error_singlechannel_signal_shapesys_k_shapesys);  
vector[2] observed_singlechannel_background_shapesys_l_shapesys = square(nominal_singlechannel_background ./ rel_error_singlechannel_background_shapesys_l_shapesys);  
vector[2] observed_secondchannel_signal_shapesys_m_shapesys = square(nominal_secondchannel_signal ./ rel_error_secondchannel_signal_shapesys_m_shapesys);  
vector[2] observed_secondchannel_background_shapesys_n_shapesys = square(nominal_secondchannel_background ./ rel_error_secondchannel_background_shapesys_n_shapesys);  
vector[2] stdev_singlechannel_k_staterror = sqrt(stdev_singlechannel_signal_staterror_k_staterror.^2) ./ (nominal_singlechannel_signal);  
vector[2] stdev_singlechannel_l_staterror = sqrt(stdev_singlechannel_background_staterror_l_staterror.^2) ./ (nominal_singlechannel_background);  
vector[2] stdev_secondchannel_m_staterror = sqrt(stdev_secondchannel_signal_staterror_m_staterror.^2) ./ (nominal_secondchannel_signal);  
vector[2] stdev_secondchannel_n_staterror = sqrt(stdev_secondchannel_background_staterror_n_staterror.^2) ./ (nominal_secondchannel_background);  
}