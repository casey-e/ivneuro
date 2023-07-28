# -*- coding: utf-8 -*-
"""
A  module with private helper functions for delta_coherence function locate in delta_coherence module.
"""

import numpy as np
import pandas as pd
from scipy.signal import coherence
from itertools import combinations
from ivneuro.continuous import calculate_sampling_rate

def calculate_delta_coherence(var_1_exp_interval, var2_exp_interval, var_1_baseline, var2_baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft):
    
    
    coherence_a = [coherence(var1, var2, sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft) for var1, var2 in zip(var_1_exp_interval, var2_exp_interval)]
    freq = [i[0] for i in coherence_a][0]
    mask= (freq >= lowest_freq) & (freq <= highest_freq) # Mask to selcet frequencies
    freq=freq[mask]
    coherence_a = np.array([i[1] for i in coherence_a]).T[mask]
    coherence_b = np.array([coherence(var1, var2, sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)[1] for var1, var2 in zip(var_1_baseline, var2_baseline)]).T[mask]
    
    diff = coherence_a-coherence_b
    diff=diff.mean(axis=1)
    
    coherence_a = coherence_a.mean(axis=1)
    coherence_b = coherence_b.mean(axis=1) 
    
    return freq, coherence_a, coherence_b, diff


def single_coherence(contvar, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft):
    
    df_list=[contvar.loc[i, :] for i in exp_interval]
    var_1_exp_interval = [df.iloc[:,0].values for df in df_list]
    var2_exp_interval = [df.iloc[:,1].values for df in df_list]
    
    df_list=[contvar.loc[i, :] for i in baseline]
    var_1_baseline = [df.iloc[:,0].values for df in df_list]
    var2_baseline = [df.iloc[:,1].values for df in df_list]
    
    return calculate_delta_coherence(var_1_exp_interval, var2_exp_interval, var_1_baseline, var2_baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft)


def multi_coherence(contvar, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft):
    
    # If sampling rate is None, calculate it using the timestamps of the index
    if sampling_rate == None:
        sampling_rate = calculate_sampling_rate(contvar.index)
    
    var_combinations = [*combinations(contvar.columns, 2)] # All posible pairs of variables
    # col_names=['{}-{}'.format(i[0],i[1]) for i in var_combinations]
    cols1, cols2=[i[0] for i in var_combinations], [i[1] for i in var_combinations] # Lists of the first and the second variable of each pair
    
    # Make all posible combinations of pairs of variables and variable names
    var_combinations = [*combinations([*range(len(contvar.columns))], 2)] # All posible pairs of variables, as column index
    varnames_combinations = [*combinations(contvar.columns, 2)] # All posible pairs of variables names
    cols1, cols2=[i[0] for i in varnames_combinations], [i[1] for i in varnames_combinations] # Lists of the first and the second variable name of each pair
    
    # Calculate coherences for each interval list and variable
    result = [single_coherence(contvar.iloc[:,[*val_pair]], exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft) for val_pair in var_combinations] #results is a list of tuples with 4 elements
    freq = result[0][0] # Get the frequencies from the first element of results    
    result = [np.array([i[j] for i in result]) for j in [1, 2, 3]] # Get the coherences for inteval, baseline and difference between intervals and and baseline, and make a numpy.arrays for each
    
    # Make dataframes
    result = (pd.DataFrame(columns=[cols1, cols2], index=freq, data=i.T) for i in result)
    return result
