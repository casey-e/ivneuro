# -*- coding: utf-8 -*-
"""
A module with delta_power_spectral function.
"""

import pandas as pd
from ivneuro.continuous import calculate_sampling_rate
from ivneuro.spectral.delta_power_spectral_core import multi_delta_power
        
            

def delta_power_spectral(contvar, exp_interval, baseline, lowest_freq = 0, highest_freq = 500, sample_subsamples =None, sampling_rate = None, nfft=2000, scaling ='spectrum'):
    """
    Calculate the power spectral at a experimental time of interest (exp_interval), at a baseline interval (baseline), and the frequency-basis normalized difference betrween the experimental interval and the baseline.
    
    
    
    This function is based on the periodogram function of scipy.signal. 
    exp_interval and baseline must be lists of slices the same lenght, when when the lists have more than one element each, the results are the averages across intervals (either power spectral at exp_interval, power spectral at baseline, or their normalized difference).
    To normalize, it uses the formula: (Power(exp_interval) - Power(baseline)) / (Power(exp_interval) + Power(baseline)) for each frequency and trial (pair of elements in interval and baseline), where Power(exp_interval) is the power at the experimental interval and Power(baseline) is the power at the baseline.

    Parameters
    ----------
    contvar : pandas.DataFrame
        Dataframe with continuous variables in each column, and timestamps as index.
    exp_interval: list
        List of slices corresponding to experimental intervals to calculate power.
    baseline : list
        List of slices corresponding to baseline intervals to calculate power.
    lowest_freq : int or float, optional
        The lowest frequency to include in the power spectral. The default is 0.
    highest_freq : int or float, optional
        The highest freq to include in the power spectral. The default is 500.
    sample_subsamples : dict or None, optional
        When a dict, key must be sample names and values must be lists of subsample names, e.g.: {'sample_A':['subsample_A1','subsample_A2'...'subsample_An'], 'sample_B':['subsample_B1','subsample_B2'...'subsample_Bn']...}.
        Power spectrals and normalized delta power spectral will be averaged across subsamples of the same sample.
        It must only be used when contvar contains multiple replicates for each observation (e.g., local field potentials recorded with multiple wires of the same tetrode). 
        If None, each column of contvar is treated as an independent sample, and therefore the results contain spectrals of every contvar column. The default is None.
    sampling_rate : int or float or None, optional
        Sampling rate of the continuous variables. If None, the sampling_rate is calculated using the inverse of the median difference between consecutive timestamps of the contvar's index. The default is None.
    nfft : int, optional
        The nfft parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 2000.
    scaling : str, optional
        The scaling parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 'spectrum'.

    Returns
    -------
    power_exp_interval: pandas.DataFrame
        Power spectral for each variable (or sample if sample_subsamples != None), at the experimental interval. Variable names (or sample names) as columns and frequency as index.
    power_baseline: pandas.DataFrame
        Power spectral for each variable (or sample if sample_subsamples != None), at the baseline interval. Variable names (or sample names) as columns and frequency as index.
    delta_power: pandas.DataFrame
        Normalized difference of power spectrals between experimental interval and baseline interval, for each variable (or sample if sample_subsamples != None). Variable names (or sample names) as columns and frequency as index.

    """
    
    # If sampling rate is None, calculate it using the timestamps of the index
    if sampling_rate == None:
        sampling_rate = calculate_sampling_rate(contvar.index)
    
    if sample_subsamples == None:
        return multi_delta_power(contvar, exp_interval, baseline, lowest_freq, highest_freq , sampling_rate, nfft, scaling)
    
    else:
        sample_list = [*sample_subsamples.keys()] # List of independent samples
        subsamples_list =[*sample_subsamples.values()] # list of replicate lists
        subsamples_list = [[i for i in subsamples if i in contvar.columns] for subsamples in subsamples_list] # List of replicates tha are actually in contvar. This allows to use a generic sample_subsamples dictionary even if there are missing replicates in some recordings
        
        # Loop over subsamples_list and calculate delta_power for those variables, get the mean and store it in all result.
        all_result=[]
        for subsamples in subsamples_list:
            result = multi_delta_power(contvar[subsamples], exp_interval, baseline, lowest_freq, highest_freq , sampling_rate, nfft, scaling) # result is a tuple that contains dataframes of the power at the interval, the power at the baseline and the normalized delta power, for each froup of subsamples
            result = [i.mean(axis=1) for i in result] # means for each dataframe of the tuple
            all_result.append(result) # List of tuples with 3 elements each (power in interval, power in baseline and delta power), each tuple corresponding to a sample
        all_result = [*zip(*all_result)] # transform to a list of three tuples, each tuple corresponding to power at interval, baseline and delta power, and each element of the tuple corrresponding to mean across replicates of each sample
        all_result = (pd.concat([*result], axis=1).set_axis(sample_list, axis=1) for result in all_result)
        return all_result
            
        


