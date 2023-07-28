# -*- coding: utf-8 -*-
"""
A module with private helper functions for peri_event_spectogram function located in spectograms
module.
"""
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from ivneuro.continuous.continuous import peh_list


def spectogram (var, sampling_rate, lower_freq=0, higher_freq=500, scaling='spectrum',nperseg=500,noverlap=400,nfft=2000):
    f, t, Sxx = spectrogram(var, sampling_rate,scaling=scaling,nperseg=nperseg,noverlap=noverlap,nfft=nfft) #run spectograms: t=time, f=frequencies, Sxx=power_values
    spect_df=pd.DataFrame(data=Sxx, index=f,columns=t).T
    spect_df=spect_df.loc[:,lower_freq:higher_freq]
    return spect_df



# Define a function to make peri-event spectograms for a single event (with multiple trials for that single event)
def single_pes(contvar, evt, lower_lim, higher_lim, lower_freq, higher_freq, sample_subsamples, sampling_rate, scaling, nperseg, noverlap, nfft, sr, lost_ts, rounding):
    
    # Make list of peri-event histograms
    df_list = peh_list(contvar, evt, lower_lim-lost_ts, higher_lim+lost_ts) 

    # Make list of indexes for peri-event spectograms (this is necessary because spectogram function will generate a dataframe with a different index and sample rate)
    idx=[(df.index[0][1], df.index[-1][1]) for df in df_list]
    idx=[(np.round(i[0] + lost_ts, rounding), np.round(i[1] - lost_ts, rounding)) for i in idx]
    idx = [np.round(np.linspace(i[0], i[1], num=int(1+(i[1]-i[0])/sr)), rounding) for i in idx]

    # Function to make peri-event histogram for a single variable
    def single_var_pes(df_list, idx, sample, subsample):
        pes=[spectogram(df[subsample], sampling_rate, lower_freq=lower_freq, higher_freq=higher_freq,scaling=scaling,nperseg=nperseg,noverlap=noverlap,nfft=nfft) for df in df_list] # List of peri-event spectograms
        evt_number = [np.array([evt_number]*len(df)) for df, evt_number in zip(pes, [*range(1,len(pes)+1)])] # List of arrays with trial number
        pes, idx, evt_number = pd.concat(pes), np.concatenate(idx), np.concatenate(evt_number)
        pes['Time']=idx
        pes['Event_number']=evt_number
        pes['Variable_name']=sample
        return pes
    # If there are not replicates of the same independent sample, make a list of peri-event spectograms, each for every variable (contvar column)
    if sample_subsamples == None:
        pes= [single_var_pes(df_list, idx, sample, sample) for sample in contvar.columns]
    
    # If there are multiple replicates for every sample, make a list of peri-event spectograms, each for every sample, with averages across replicates of the same sample
    else:
        sample_list = [*sample_subsamples.keys()] # List of independent samples
        subsamples_list =[*sample_subsamples.values()] # list of replicate lists
        subsamples_list = [[i for i in subsamples if i in contvar.columns] for subsamples in subsamples_list] # List of replicates tha are actually in contvar. This allows to use a generic sample_subsamples dictionary even if there are missing replicates in some recordings
        pes = [pd.concat([single_var_pes(df_list, idx, sample, subsample) for subsample in subsamples]).groupby(['Time', 'Event_number','Variable_name'], as_index=False).mean() for sample,subsamples in zip(sample_list, subsamples_list)]
        
    pes=pd.concat(pes) # Return a peri-event spectogram dataframe with frequencies as columns, plus a column for Time around the event, the trial number (Event number) and Variable name
    return pes