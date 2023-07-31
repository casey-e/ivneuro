# -*- coding: utf-8 -*-
"""
A module with private helper functions for delta_power_spectral function located in delta_power_spectral
module.

"""

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def calculate_delta_power(var_exp_interval, var_baseline, lowest_freq, highest_freq, sampling_rate, nfft, scaling):
    
    
    power_a = [periodogram(var, sampling_rate, nfft=nfft, scaling=scaling) for var in var_exp_interval]
    freq = [i[0] for i in power_a][0]
    mask= (freq >= lowest_freq) & (freq <= highest_freq) # Mask to selcet frequencies
    freq=freq[mask]
    power_a = np.array([i[1] for i in power_a]).T[mask]
    power_b = np.array([periodogram(var, sampling_rate, nfft=nfft, scaling =scaling)[1] for var in var_baseline]).T[mask]
    
    
    delta = (power_a-power_b)/(power_a+power_b)
    delta=delta.mean(axis=1)
    
    power_a = power_a.mean(axis=1)
    power_b = power_b.mean(axis=1) 
    
    return freq, power_a, power_b, delta
    # return freq, coherence_a
    
def single_delta_power(contvar, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nfft, scaling):
    
    var_exp_interval=[contvar.loc[i].values for i in exp_interval]
    var_baseline=[contvar.loc[i].values for i in baseline]
    
    return calculate_delta_power(var_exp_interval, var_baseline, lowest_freq, highest_freq, sampling_rate, nfft, scaling)


def multi_delta_power(contvar, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nfft, scaling):
    
    # Calculate power for each interval list and variable
    result = [single_delta_power(contvar[var], exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nfft, scaling) for var in contvar.columns] # result is a list of tuples with 4 elements per tuple: frequencies, power of interval, power of baseline and difference
    freq = result[0][0] # Get the frequencies from the first element of results  
    result = [np.array([i[j] for i in result]) for j in [1, 2, 3]] # Get the power spectrals for inteval, baseline and difference between intervals and and baseline, and make a numpy.arrays for each
    # Make dataframes
    result = (pd.DataFrame(columns=contvar.columns, index=freq, data=i.T) for i in result)
    
    return result