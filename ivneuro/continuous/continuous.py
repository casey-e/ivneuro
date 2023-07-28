# -*- coding: utf-8 -*-
"""

User-facing functions for analysis of continuous variables.

"""

import numpy as np
import pandas as pd
from ivneuro.events.utils import significant_decimal_positions
from ivneuro.continuous.PeriEventHistogram import PeriEventHistogram

def calculate_sampling_period(timestamps):
    """
    Calculate the sampling period from an array of timestamps

    Parameters
    ----------
    timestamps : numpy.ndarray
        One dimensional array with timestamps.

    Returns
    -------
    float
        Sampling period.
    """

    return np.median(np.diff(timestamps))


def calculate_sampling_rate(timestamps):
    """
    Calculate sampling rate from an array of timestamps

    Parameters
    ----------
    timestamps : numpy.ndarray
        One dimensional array with timestamps.

    Returns
    -------
    sampling_rate : float
        Sampling rate.

    """
    
    sampling_rate = 1/calculate_sampling_period(timestamps) # Calculate the sample rate as the inverse of sampling period
    sampling_rate=np.round(sampling_rate, significant_decimal_positions(sampling_rate))
    return sampling_rate

def peh_list(contvar, evt, lower_lim, higher_lim):
    """
    Make list of peri-event histograms 

    Parameters
    ----------
    contvar : pandas.DataFrame
        Dataframe with continuous variables in each column, and timestamps as index.
    evt : one dimensional numpy.ndarray or list
        Timestamps of reference event.
    lower_lim : numeric
        Lower time limit of the peri-event histogram.
    higher_lim : numeric
        Higher time limit of the peri-event histogram.

    Returns
    -------
    peh : list
        Lst of Dataframes of peri-event histograms, each with original continuous variables as columns, and multi-index with trial number and peri-event time.

    """
    # Calculate rounding, a variable that will be used to round the peri-event time, used as index in the returned dataframe
    # This rounding is necessary to avoid artifacts cased by switching from decimal to binary numeric system in vectorized calculus
    sampling_period = calculate_sampling_period(contvar.index) # Use media of delta timestamp to estimate the sample rate
    rounding=significant_decimal_positions(sampling_period) # This formula ensure to get enough decimal positions while discarding decimals values caused by binary to decimal system transformations
       
    #Use contvar.iloc[np.searchsorted(contvar.index, evt),].index to get the list of indexes of contvar dataframe, and 
    #contvar.loc[(i+lower_lim):(i+higher_lim)] to slice contvar dataframe around each event timestamp
    peh=[contvar.loc[(i+lower_lim):(i+higher_lim)] for i in contvar.iloc[np.searchsorted(contvar.index, evt),].index] # list of contvar slices
    index=[np.round((contvar.loc[(i+lower_lim):(i+higher_lim)].index - i), rounding) for i in contvar.iloc[np.searchsorted(contvar.index, evt, 'right'),].index] # list of index slices
    
    peh=[df.set_index([np.array([evt_number]*len(df)),idx]) for df, idx, evt_number in zip(peh, index, [*range(1,len(peh)+1)])] # list of peri event histogram dataframes with multiindex of event number and peri-event time
    
    return peh

def single_peh(contvar, evt, lower_lim, higher_lim):
    """
    Make peri-event histograms for a single event (with multiple trials)

    Parameters
    ----------
    contvar : pandas.DataFrame
        Dataframe with continuous variables in each column, and timestamps as index.
    evt : one dimensional numpy array or list
        Timestamps of reference event.
    lower_lim : numeric
        Lower time limit of the peri-event histogram.
    higher_lim : numeric
        Higher time limit of the peri-event histogram.

    Returns
    -------
    peh : pandas.DataFrame
        Dataframe of peri-event histograms with original continuous variables as columns, and multi-index with trial number and peri-event time.

    """
    result = peh_list(contvar, evt, lower_lim, higher_lim) # list of peri event histogram dataframes with multiindex of event number and peri-event time
    
    result=pd.concat(result)
    return result

def peh(contvar, evt, lower_lim, higher_lim, return_DataFrame = False):
    
    """
    Make peri-event histograms 

    Parameters
    ----------
    contvar : pandas.DataFrame
        Dataframe with continuous variables in each column, and timestamps as index.
    evt : one-dimensional numpy.ndarray, list or dict
        Timestamps of a single reference event if evt is a one-dimesional np.ndarray or a list. If multiple events are analized, evt must be a dict with event names as keys and timestamps as values, for every reference event. 
        Dict values can be either one dimensional numpy arrays or lists of floats.
    lower_lim : int or float
        Lower time limit of the peri-event histogram.
    higher_lim : int or float
        Higher time limit of the peri-event histogram.

    Returns
    -------
    peh : pandas.DataFrame
        Dataframe of peri-event histograms with original continuous variables as columns, and multi-index with event names, trial number and peri-event time.

    """
    
    if type(evt) == list or type (evt) == np.ndarray:
        evt= {'Event':evt}
    elif type(evt) != dict:
        raise(TypeError('evt must be either numpy.ndarray, list or dictionary'))
    
    result=[(single_peh(contvar, evt[event], lower_lim, higher_lim), event) for event in evt] # Apply single_peh function to make a peri-event histograms dataframe for each event
    result=[df.set_index([np.array([event]*len(df)),df.index.get_level_values(0),df.index.get_level_values(1)]) for df, event in result] #Add index level with event names
    result= pd.concat(result)
    result.index.names=['Event_name','Event_number', 'Time']
    if not return_DataFrame:
        result = PeriEventHistogram(result)
    return result
    

