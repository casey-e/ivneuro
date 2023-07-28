# -*- coding: utf-8 -*-
"""
A module with delta_coherence function.
"""


from itertools import chain
from ivneuro.spectral.delta_coherence_core import multi_coherence




def delta_coherence(contvar, exp_interval, baseline, lowest_freq = 0, highest_freq = 500, sample_subsamples =None, sampling_rate = None, nperseg=500,noverlap=400,nfft=2000):
    """
    Calculate the coherence between 2 signals at a experimental interval of time of interest (exp_interval), at a baseline interval of time (baseline), and the difference betrween the spectral at the experimental interval and the baseline.
    
    
    
    This function is based on the coherence function of scipy.signal, wich uses Welch's method. 
    exp_interval and baseline must be lists of slices of the same lenght, when when the lists have more than one element each, the results are the averages across intervals (either coherence at exp_interval, coherence at baseline, or their difference).   
    To calculate the difference of coherence between experimental interval and baseline, the function subtracts the coherence at baseline from the coherence at exp_interval, for each frequency and trial (pair of elements in interval and baseline).

    Parameters
    ----------
    contvar : pandas.DataFrame
        Dataframe with continuous variables in each column, and timestamps as index.
    exp_interval: list
        List of slices corresponding to experimental intervals to calculate coherence.
    baseline : list
        List of slices corresponding to baseline intervals to calculate coherence.
    lowest_freq : int or float, optional
        The lowest frequency to include in the coherence spectral. The default is 0.
    highest_freq : int or float, optional
        The highest freq to include in the coherence spectral. The default is 500.
    sample_subsamples : dict or None, optional
        When a dict, key must be sample names and values must be lists of subsample names, e.g.: {'sample_A':['subsample_A1','subsample_A2'...'subsample_An'], 'sample_B':['subsample_B1','subsample_B2'...'subsample_Bn']...}.
        Coherences and coeherence differences will be averaged across pairs of subsamples corresponding to the same samples pair.
        It must only be used when contvar contains multiple replicates for each observation (e.g., local field potentials recorded with multiple wires of the same tetrode). 
        If None, each column of contvar is treated as an independent sample, and therefore the results contain coherences for every pair of contvar columns. The default is None.
    sampling_rate : int or float or None, optional
        Sampling rate of the continuous variables, wich is used in signal.coherence() function of the scipy package. If None, the sampling_rate is calculated using the inverse of the median difference between consecutive timestamps of the contvar's index. The default is None.
    nperseg : int, optional
        The nperseg parameter to enter to signal.coherence() function of the scipy package. Refer to scipy.signal.coherence in scipy manual for more information. The default is 500.
    noverlap : int, optional
        The noverlap parameter to enter to signal.coherence() function of the scipy package. Refer to scipy.signal.coherence in scipy manual for more information. The default is 400.
    nfft : int, optional
        The nfft parameter to enter to signal.coherence() function of the scipy package. Refer to scipy.signal.coherence in scipy manual for more information. The default is 2000.
 

    Returns
    -------
    coherence_exp_interval: pandas.DataFrame
        Coherence for each pair of variables (or pair of samples if sample_subsamples != None), at the experimental interval. Variable names (or sample names) as 2-levels columns (one level for each member of the pair) and frequency as index.
    power_baseline: pandas.DataFrame
        Coherence for each pair of variables (or pair of samples if sample_subsamples != None), at the baseline interval. Variable names (or sample names) as 2-levels columns (one level for each member of the pair) and frequency as index.
    delta_coherence: pandas.DataFrame
        Difference of coherence between experimental interval and baseline interval, for each pair of variables (or pair of samples if sample_subsamples != None). Variable names (or sample names) as 2-levels columns (one level for each member of the pair) and frequency as index.

    """
    
    # If there are not subsamples, just run multi_coherence
    if sample_subsamples == None:
        return multi_coherence(contvar, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft)
    
    # If there are subsamples, reorder and rename variables, then run delta_coherence and finally calculate averages across replicates of the same pair of variables
    else:
        # Reorder variables as in sample_subsamples dictionary to prevent to have replicates flipped (for example A1B1, A2B1, A1B2, B2A1)
        col_list= [var_name for var_name in [*chain(*sample_subsamples.values())] if var_name in contvar.columns] # List of variable names, ordered as in sample_subsamples dictionary
        new_data=contvar[col_list]
        
        # Replace replicate by independent observation in column names
        replacing={name:new_name for name in new_data.columns for new_name, old_names in sample_subsamples.items() if name in old_names}
        new_data=new_data.rename(columns=replacing) # Now there are columns with the same name, which allow to later group them and calculate the mean
        
        result=multi_coherence(new_data, exp_interval, baseline, lowest_freq, highest_freq, sampling_rate, nperseg, noverlap, nfft)
        result =(df.groupby(level=[0,1], axis=1).mean() for df in result)
        return result
        