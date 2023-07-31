# -*- coding: utf-8 -*-
"""
A module with functions for extracting data from nex5 files.
"""

import numpy as np
import pandas as pd
from . import nexfile

reader = nexfile.Reader(useNumpy=True)

def make_timestamps_Nex_cont(var):
    time_stamps=[]
    for i in range(len(var['FragmentIndexes'])): #iterate thorugh fragments
        first_ts=var['FragmentTimestamps'][i] # get the value of the first timestamp of the fragment
        ts=[(first_ts+x*var['Header']['SamplingRate']**(-1)) for x in range (var['FragmentCounts'][i])]
        time_stamps=time_stamps+ts # concatenate list of timestamps
    assert(len(time_stamps)==len(var['ContinuousValues'])) # evaluate that the number of timestaps is the same than the number of continuous values
    return time_stamps

def pull_fp(fileData, FP_of_interest=[]):
    """
    Extract field potential variables from a Nex5 file data, previously created using reader.ReadNex5File(<filename>).
    
    It assumes that "FP" will be in the name of every field potential variable, which is true for Nex files whose original variables names have not been modified.
    Therefore, restults of this function might be incorrect for files whose variable names have been modified.
    It also assumes all field potential variables have the same timestamps and sample rate.

    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
        
    FP_of_interest : list, optional
        List of names of field potentials to extract. If None, the function returns None; if an empty list, all the field potentials are extracted. The default is [].

    Returns
    Dataframe with field potentials (in mV) in each column and timestaps as index.
    
    Sampling rate of local field potentials (float).

    """
    if FP_of_interest == None:
        fp = None
        sampling_rate = None
    else:
        if len(FP_of_interest) > 0:
            fp= {variable['Header']['Name']:variable['ContinuousValues'] for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and variable['Header']['Name'] in FP_of_interest}
            first_fp=next((variable for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and variable['Header']['Name'] in FP_of_interest))

        else:
            fp= {variable['Header']['Name']:variable['ContinuousValues'] for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' in variable['Header']['Name']}
            first_fp=next((variable for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' in variable['Header']['Name']))
           
        time_stamps= make_timestamps_Nex_cont(first_fp) # Make timestamps using the firts field potential
       
        sampling_rate= first_fp['Header']['SamplingRate'] # Get sampling rate
         
        fp=pd.DataFrame(fp, index=time_stamps) # Make dataframe with signal from every channel (LFP in mV) and timestamps as index
        fp.index=np.round(fp.index, 3) # Round timestamps to 3 decimals
            
    return fp, sampling_rate


def pull_events(fileData, events_of_interest=[]):
    """
    
    Extract events from a Nex5 file data, previously created using reader.ReadNex5File(<filename>).

    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
    events_of_interest : list, optional
        List of names of events to extract. If None, the function returns None; if an empty list, all the events are extracted. The default is [].

    Returns
    -------
    Dictionary with event names as keys and list of timestamps as values.

    """
    if events_of_interest == None:
        events = None
    elif len(events_of_interest) > 0:
        events = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 1 and variable['Header']['Name'] in events_of_interest}
    else:
       events = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 1}

    return events


def pull_continuous(fileData, continuous_of_interest=[]):
    """
    Extract all continuous variables, except for field potential variables, from a Nex5 file data previously created using reader.ReadNex5File(<filename>).
    
    
    It assumes that "FP" will be in the name of every field potential variable, which is true for Nex files whose original variables names have not been modified.
    Therefore, restults of this function might be incorrect for files whose variable names have been modified.


    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
    continuous_of_interest : list, optional
        List of names of continuous to extract. If None, the function returns None; if an empty list, all non-field potential continuous variables are extracted. The default is [].

    Returns
    -------
    List of pandas DataFrame, each corresponding to a continuous variable values and timestaps as index.

    """
    if continuous_of_interest == None:
        continuous = None
    elif len(continuous_of_interest) > 0:
        continuous = [pd.DataFrame({variable['Header']['Name']:variable['ContinuousValues']}, index=variable['FragmentTimestamps']) for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' not in variable['Header']['Name'] and variable['Header']['Name'] in continuous_of_interest]
    else:
        continuous = [pd.DataFrame({variable['Header']['Name']:variable['ContinuousValues']}, index=variable['FragmentTimestamps']) for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' not in variable['Header']['Name']]

    return continuous

def pull_neurons(fileData, neurons_of_interest=[]):
    """
    Extract neurons from a Nex5 file data, previously created using reader.ReadNex5File(<filename>).

    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
    neurons_of_interest : list, optional
        List of names of events to extract. If None, the function returns None; if an empty list, all the neurons are extracted. The default is [].

    Returns
    -------
    Dictionary with neuron names as keys and list of timestamps as values.

    """
    if neurons_of_interest == None:
        neurons = None
    elif len(neurons_of_interest) > 0:
        neurons = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 0 and variable['Header']['Name'] in neurons_of_interest}
    else:
       neurons = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 0}

    return neurons

def pull_markers(fileData, markers_of_interest=[]):
    """
    Extract markers from a Nex5 file data, previously created using reader.ReadNex5File(<filename>).

    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
    markers_of_interest : list, optional
        List of names of markers to extract. If None, the function returns None; if an empty list, all the markers are extracted. The default is [].

    Returns
    -------
    Dictionary with markers names as keys and list of timestamps as values.

    """
    if markers_of_interest == None:
        markers = None
    elif len(markers_of_interest) > 0:
        markers = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 6 and variable['Header']['Name'] in markers_of_interest}
    else:
       markers = {variable['Header']['Name']:variable['Timestamps'] for  variable in fileData['Variables'] if variable['Header']['Type'] == 6}

    return markers

def pull_centroids(fileData, centroids_of_interest=[]):
    """
    Extract field potential variables from a Nex5 file data, previously created using reader.ReadNex5File(<filename>).
    
    It assumes that "centroid" (case insensitive) will be in the name of every centroid variable, which is true for Nex files whose original variables names have not been modified.
    It also exclude field potential variables that contain the string "FP" in their names. Therefore, restults of this function might be incorrect for files whose variable names have been modified.
    It also assumes that all centroids have the same timestamps.
    
    Parameters
    ----------
    fileData : dict
        Data extracted from a nex5 file using reader.ReadNex5File(<filename>).
    centroids_of_interest : list, optional
        List of names of centroids to extract. If None, the function returns None; if an empty list, all centroid variables are extracted. The default is [].

    Returns
    -------
    Dataframe with centroids in each column and timestaps as index.

    """
    if centroids_of_interest == None:
        continuous = None
    elif len(centroids_of_interest) > 0:
        centroids = [pd.DataFrame({variable['Header']['Name']:variable['ContinuousValues']}, index=variable['FragmentTimestamps']) for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' not in variable['Header']['Name'] and 'centroid' in variable['Header']['Name'].lower() and variable['Header']['Name'] in continuous_of_interest]
    else:
        centroids = [pd.DataFrame({variable['Header']['Name']:variable['ContinuousValues']}, index=variable['FragmentTimestamps']) for variable in fileData['Variables'] if variable['Header']['Type'] == 5 and 'FP' not in variable['Header']['Name'] and 'centroid' in variable['Header']['Name'].lower()]
    
    centroids=pd.concat(centroids, axis=1)
    
    return centroids

