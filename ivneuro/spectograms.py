# -*- coding: utf-8 -*-
"""
User-facing functions to calculate, normalyze and plot peri-event spectograms analyses.
This module also contains PeriEventSpectogram class. Peri-event spectograms analysis can be higly 
demanding, depending on the size and amount of continuous variables, the frequencies analysed, the
amount of reference events and number of trials of each of event. PeriEventSpectogram class aims to
facilitate processing, interpretation and visualization of peri-event spectograms without running the
analysis again and is returned by default by ivneuro.spectral.peri_event_spectogram function.
"""


import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from itertools import product
from .spectograms_core import single_pes
from .utils import significant_decimal_positions
from .continuous import calculate_sampling_rate



def peri_event_spectogram (contvar, evt, lower_lim, higher_lim, lower_freq=0, higher_freq=500, sample_subsamples =None, return_DataFrame = False, sampling_rate = None, scaling='spectrum',nperseg=500,noverlap=400,nfft=2000):
    """
    
    Calculate peri-event spectograms.
    
    The algorithm is based on the spectogram() function of the signal processing module of the scipyi package (scipy.signal.spectogram()), and is optimized to calculate peri-event spectograms from multiple continuous variables and/or multiple events, 
    with multiple trials of each event. 

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
    lower_freq : int or float, optional
        The lowest frequency to include in the spectogram. The default is 0.
    higher_freq : int or float, optional
        The highest freq to include in the spectogram. The default is 500.
    sample_subsamples : dict or None, optional
        When a dict, key must be sample names and values must be lists of subsample names, e.g.: {'sample_A':['subsample_A1','subsample_A2'...'subsample_An'], 'sample_B':['subsample_B1','subsample_B2'...'subsample_Bn']...}.
        Peri-event spectograms will be averaged across subsamples of the same sample, per each time and trial of each event, and the Variable name will be the sample name, intead of the column name of contvar.
        It must only be used when contvar contains multiple replicates for each observation (e.g., local field potentials recorded with multiple wires of the same tetrode). 
        If None, each column of contvar is treated as an independent sample, and therefore the result contains peri-event spectograms of every contvar column. The default is None.
    return_DataFrame : Boolean, optional
        If True, a pandas.DataFrame is returnes. If false, a PeriEventHistogram object is returned. The default is False.
    sampling_rate : int, float or None, optional
        Sampling rate of the continuous value. If None, the sampling_rate is calculated using the inverse of the median difference between consecutive timestamps of the contvar's index. The default is None.
    scaling : str, optional
        The scaling parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 'spectrum'.
    nperseg : int, optional
        The nperseg parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 500.
    noverlap : int, optional
        The noverlap parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 400.
    nfft : int, optional
        The nfft parameter to enter to signal.spectrogram() function of the scipy package. Refer to scipy.signal.spectogram in scipy manual for more information. The default is 2000.

    Raises
    ------
    TypeError
        If evt type is neither np.ndarray, list, dict.

    Returns
    -------
    pes : pandas.DataFrame or PeriEventSpectogram
        If return_DataFrame == True, multi-index pandas.DataFrame with original continuous variable names, event names, event trial number and peri-event time as index, frequencies as columns and power values as data.
        If return_DataFrame == False, PeriEventSpectogram object with the already described pandas.DataFrame as data, and normalization = 'None'. 
    
    
    
    
    Examples
    --------
    
    Create events and signals.
    
    >>> import ivneuro as ivn
    >>> import pandas as pd
    >>> # Create events: burst and control
    >>> burst = [*range(30,300, 30)]
    >>> control = [*range(45,300, 30)]
    >>> events = {'burst':burst,'control':control}
    >>> # Generate signals
    >>> signal1 = ivn.generate_signal(300, burst, 30, burst_amplitude=0.06)
    >>> signal2 = ivn.generate_signal(300, burst, 32, burst_amplitude=0.13)
    >>> signal3 = ivn.generate_signal(300, burst, 80, burst_amplitude=0.05)
    >>> signals = pd.concat([signal1, signal2, signal3], axis = 1)
    
    Peri-event spectograms for a single variable and a single event (with multiple trials).
    
    >>> # Calculate peri-event spectogram from 10 sec before to 10 seconds after each event
    >>> pes = ivn.peri_event_spectogram(signal1, burst, -10, 10)
    >>> type(pes)
    ivneuro.spectograms.PeriEventSpectogram
    >>> pes.data
                                                    0.0    ...         500.0
    Variable_name Event_name Event_number Time             ...              
    Signal 30Hz   Event      1            -10.0  0.003922  ...  7.248012e-08
                                          -9.9   0.000142  ...  9.831598e-08
                                          -9.8   0.020257  ...  2.587770e-06
                                          -9.7   0.005431  ...  5.538603e-06
                                          -9.6   0.002156  ...  5.956565e-06
                                                      ...  ...           ...
                             9             9.6   0.002953  ...  4.089984e-06
                                           9.7   0.009216  ...  1.163104e-05
                                           9.8   0.001603  ...  5.611661e-06
                                           9.9   0.000025  ...  9.778874e-07
                                           10.0  0.002685  ...  1.019278e-06                                    
    [1809 rows x 1001 columns]
    
    
    Return a pandas.DataFrame instead of a ivneuro.spectograms.PeriEventSpectogram.
    
    >>> pes = ivn.peri_event_spectogram(signal1, burst, -10, 10, return_DataFrame = True)
    >>> type(pes)
    pandas.core.frame.DataFrame
    >>> pes
                                                    0.0    ...         500.0
    Variable_name Event_name Event_number Time             ...              
    Signal 30Hz   Event      1            -10.0  0.003922  ...  7.248012e-08
                                          -9.9   0.000142  ...  9.831598e-08
                                          -9.8   0.020257  ...  2.587770e-06
                                          -9.7   0.005431  ...  5.538603e-06
                                          -9.6   0.002156  ...  5.956565e-06
                                                      ...  ...           ...
                             9             9.6   0.002953  ...  4.089984e-06
                                           9.7   0.009216  ...  1.163104e-05
                                           9.8   0.001603  ...  5.611661e-06
                                           9.9   0.000025  ...  9.778874e-07
                                           10.0  0.002685  ...  1.019278e-06                                      
    [1809 rows x 1001 columns]
    
    
    Peri-event spectogram for 2 events and 3 signals, set frequency limits.
    
    >>> pes = ivn.peri_event_spectogram(signals, events, -10, 10, lower_freq = 30, higher_freq=150)
    >>> pes.event_names
    ['burst', 'control']
    >>> pes.variable_names
    ['Signal 30Hz', 'Signal 32Hz', 'Signal 80Hz']
    >>> pes.frequencies[0], pes.frequencies[-1]
    (30.0, 150.0)
    
    
    Set signal1 and signal2 as replicates of sample1, and signal3 as sample3.
    
    >>> # Make dictionary to group replicates of each sample
    >>> sample_subsamples = {'sample1' : ['Signal 30Hz', 'Signal 32Hz'], 'sample2':['Signal 80Hz']}
    >>> # Calculate peri-event spectogram
    >>> pes = ivneuro.peri_event_spectogram(signals, events, -10, 10, higher_freq=100, sample_subsamples = sample_subsamples)    
    >>> # Print variables
    >>> pes.variable_names
    ['sample1', 'sample2']
    

    """
    
    
    # If sampling rate is None, calculate it using the timestamps of the index
    if sampling_rate == None:
        sampling_rate = calculate_sampling_rate(contvar.index)

    #Calculate variables that will be necessary to make peri-event spectogram index
    sr = (nperseg - noverlap)/1000 # Spectogram sampling rate
    lost_ts=nperseg/(2*1000) # Timestamps that are lost due to the windowing of spectogram function
    rounding=significant_decimal_positions(sr) # Amount of significant decimal positions to make index for peri-event spectograms
    
    # Define current_single_pes by setting the parameters of single_pes to the its arguments
    current_single_pes=partial(single_pes, lower_lim=lower_lim, higher_lim=higher_lim, lower_freq=lower_freq, higher_freq=higher_freq, sample_subsamples=sample_subsamples, sampling_rate=sampling_rate, scaling=scaling, nperseg=nperseg, noverlap=noverlap, nfft=nfft, sr=sr, lost_ts=lost_ts, rounding=rounding)
  
    # Run single_pes or multi_pes, depending on if evt is numpy.ndarray/list or dictionary. Then set non-power variables as index
    
    if type(evt) == list or type(evt)==np.ndarray:
        pes=current_single_pes(contvar, evt)
        pes['Event_name']='Event'
    
    elif type (evt) ==dict:
        # Use lists conmprehension to loop through events applying single_pes and adding the event name  
        pes = [current_single_pes(contvar, event_ts) for event_ts in evt.values()]
        evt_arrays = [np.array([event]*len(df)) for df, event in zip(pes, evt)] # List of numpy.ndarrays with event names
        pes, evt_arrays = pd.concat(pes), np.concatenate(evt_arrays)
        pes['Event_name'] = evt_arrays # Add event names to peri-event spectogram

    else:
        raise TypeError ('evt must be either a list of event timestamps or a dictionary with event names as keys and list of event timestamps as values')
    
    pes.set_index(['Variable_name','Event_name','Event_number','Time'], inplace=True)
    pes.sort_index(level=[0, 1, 2, 3], inplace=True) # Since the algirithm loops over variables before than over events types, indexes need to be sorted
    if not return_DataFrame:
        pes = PeriEventSpectogram(pes, norm='None')
    return pes

  

def normalize_pes(pes, baseline=None, method='Condition_average'):
    """
    Normalize peri-event-histogram using the formula (X - Mean(baseline)) / Mean(baseline) for each frequency, where X is every value and Mean(baseline) is the mean value of a baseline.
    
    While using decibels or logarithmic scale instead of power (V**2) can helph to displaying spectograms, usualy this is not enough to visualyze variations in both low and high frequencies.
    Normalizing the data to a baseline is a simple and straingforward way to visualyze variation in all frequencies at the same scale.

    Parameters
    ----------
    pes : pandas.DataFrame
        Multi-index pandas.DataFrame as returned by peri_event_spectogram() function, with original continuous variable names, event names, event trial number and peri-event time as index, frequencies as columns and power values as data.
    baseline : tuple, optional
        lowest and highest time limits of the baseline, or None. If None, baseline = (lowest_lim, highest_lim) of the argument passed to pes. The default is None.
    method : str, optional
        Method used to nromalize.
        "Condition_average": Mean(baseline) is calculated across all trials of all events and used to normalize the values of the peri-event spectogram.
        "Condition_specific": Mean(baseline) is calculated across all trials of each event, and used to normalize the data that specific event. This can be convenient when baselines differ between events.
        "Trial_specific": Mean(baseline) is calculated for each trial of each event, and used to normalize only that trial.
        The default is 'Condition_average'.

    Raises
    ------
    TypeError
        If baseline is neither a tuple or None.
    NameError
        If method is neither "Condition_average", "Condition_specific" or "Trial_specific".

    Returns
    -------
    normalized_df : pandas.DataFrame
        DataFrame with normalized data.
    
    
    Examples
    --------
    
    Create events and signals, and make peri-event spectograms DataFrame.
    
    >>> import ivneuro as ivn
    >>> import pandas as pd
    >>> # Create events: burst and control
    >>> burst = [*range(30,300, 30)]
    >>> control = [*range(45,300, 30)]
    >>> events = {'burst':burst,'control':control}
    >>> # Generate signals
    >>> signal1 = ivn.generate_signal(300, burst, 30, burst_amplitude=0.06)
    >>> signal2 = ivn.generate_signal(300, burst, 32, burst_amplitude=0.13)
    >>> signal3 = ivn.generate_signal(300, burst, 80, burst_amplitude=0.05)
    >>> signals = pd.concat([signal1, signal2, signal3], axis = 1)
    >>> pes = ivn.peri_event_spectogram(signals, events, -10, 10, return_DataFrame = True)
    
    Use function to normalize.
    
    >>> ivn.normalize_pes(pes)
                                                    0.0    ...     500.0
    Variable_name Event_name Event_number Time             ...          
    Signal 30Hz   burst      1            -10.0  0.253770  ... -0.979913
                                          -9.9  -0.954578  ... -0.972753
                                          -9.8   5.476162  ... -0.282843
                                          -9.7   0.736362  ...  0.534932
                                          -9.6  -0.310721  ...  0.650763
                                                      ...  ...       ...
    Signal 80Hz   control    9             9.6  -0.258766  ...  0.130080
                                           9.7  -0.689950  ... -0.864958
                                           9.8  -0.795989  ... -0.800983
                                           9.9  -0.800038  ...  0.999065
                                           10.0 -0.859353  ...  4.566640
    [10854 rows x 1001 columns]


    Normalize to a baseline.
    
    >>> ivn.normalize_pes(pes, baseline = (-10,-5))
                                                    0.0    ...     500.0
    Variable_name Event_name Event_number Time             ...          
    Signal 30Hz   burst      1            -10.0  0.284930  ... -0.980264
                                          -9.9  -0.953449  ... -0.973230
                                          -9.8   5.637114  ... -0.295378
                                          -9.7   0.779516  ...  0.508103
                                          -9.6  -0.293591  ...  0.621910
                                                      ...  ...       ...
    Signal 80Hz   control    9             9.6  -0.240397  ...  0.110194
                                           9.7  -0.682266  ... -0.867334
                                           9.8  -0.790933  ... -0.804485
                                           9.9  -0.795083  ...  0.963888
                                           10.0 -0.855867  ...  4.468686
    [10854 rows x 1001 columns]  
    
    
    Normalize using "Trial_specific" method.
    
    >>> ivn.normalize_pes(pes, method = "Trial_specific")
                                                    0.0    ...     500.0
    Variable_name Event_name Event_number Time             ...          
    Signal 30Hz   burst      1            -10.0  0.322110  ... -0.976902
                                          -9.9  -0.952102  ... -0.968668
                                          -9.8   5.829160  ... -0.175315
                                          -9.7   0.831006  ...  0.765074
                                          -9.6  -0.273151  ...  0.898273
                                                      ...  ...       ...
    Signal 80Hz   control    9             9.6  -0.211468  ...  0.246262
                                           9.7  -0.670165  ... -0.851075
                                           9.8  -0.782971  ... -0.780522
                                           9.9  -0.787279  ...  1.204586
                                           10.0 -0.850378  ...  5.138939
    [10854 rows x 1001 columns]
    

    """
    
    if baseline == None:
        bas=slice(None)
    elif type(baseline) == tuple:
        bas = slice(baseline[0], baseline[1])
    else:
        raise TypeError('Baseline must be either tuple or None')
    
         
    if method =='Condition_average': # In this case avg is calculated as the mean of the baseline across all trials
        normalized_df = [pes.loc[FP,:] for FP in pes.index.levels[0]]
        avg = [df.loc[slice(None), slice(None), bas,:].mean(axis=0) for df in normalized_df]
        normalized_df = [df.subtract(avg_i,axis=1).divide(avg_i, axis=1) for df, avg_i in zip(normalized_df, avg)]
        if len(pes.index.levels[1]) <= 1:
            print('Note that for single event peri-event histograms, "Condition_average" and "Condition_specific" methods return the same result')
        
    elif method =='Condition_specific': # In this case avg is calculates as the mean of the baseline across trials of each event, this means the baseline is different for each event type
        normalized_df=[pes.loc[(FP, event),:] for FP in pes.index.levels[0] for event in pes.index.levels[1]]
        avg = [df.loc[slice(None),bas,:].mean(axis=0) for df in normalized_df]
        normalized_df = [df.subtract(avg_i,axis=1).divide(avg_i, axis=1) for df, avg_i in zip(normalized_df, avg)]
        if len(pes.index.levels[1]) <= 1:
            print('Note that for single event peri-event histograms, "Condition_average" and "Condition_specific" methods return the same result')
        
    elif method =='Trial_specific': # In this case avg is calculated as the mean of the baseline of each trial, this means that there is a baseline for each trial  
        normalized_df=[pes.loc[(FP, event),:] for FP in pes.index.levels[0] for event in pes.index.levels[1]]
        normalized_df = [df.loc[trial,:] for df in normalized_df for trial in set(df.index.get_level_values(0))]
        avg = [df.loc[bas,:].mean(axis=0) for df in normalized_df]
        normalized_df = [df.subtract(avg_i,axis=1).divide(avg_i, axis=1) for df, avg_i in zip(normalized_df, avg)]
            
    else:
        raise NameError ('"method" name must be either "Condition_average", "Condition_specific" or "Trial_specific", but "{}" was found.'.format(method))
    
    normalized_df=pd.concat(normalized_df)
    normalized_df.set_index(pes.index, inplace=True)
    # normalized_df = PeriEventSpectogram(normalized_df, norm=method)
    return normalized_df

def plot_pes(pes, zero_centered=True, aspect=1):
    """
    Plot peri-event spectograms, with each variable in a column and each event name in a row.

    Parameters
    ----------
    pes : pandas.DataFrame
        Multi-index pandas.DataFrame as returned by peri_event_spectogram() nor normalize_pes() function, with original continuous variable names, event names, event trial number and peri-event time as index, frequencies as columns and power values as data.
    zero_centered : boolean, optional
        If True, data is centered to zero. The default is True.
    aspect : float, optional
        The y/x ratio of the axes aspect. The default is 1.

    Returns
    -------
    None.
    
    Examples
    --------
    
    Create events and signals, and make peri-event spectograms DataFrame.
    
    >>> import ivneuro as ivn
    >>> import pandas as pd
    >>> # Create events: burst and control
    >>> burst = [*range(30,300, 30)]
    >>> control = [*range(45,300, 30)]
    >>> events = {'burst':burst,'control':control}
    >>> # Generate signals
    >>> signal1 = ivn.generate_signal(300, burst, 30, burst_amplitude=0.06)
    >>> signal2 = ivn.generate_signal(300, burst, 32, burst_amplitude=0.13)
    >>> signal3 = ivn.generate_signal(300, burst, 80, burst_amplitude=0.05)
    >>> signals = pd.concat([signal1, signal2, signal3], axis = 1)
    >>> pes = ivn.peri_event_spectogram(signals, events, -10, 10, higher_freq=90, return_DataFrame = True)
    
    Use function to plot peri-event spectogram.
    
    >>> ivn.plot_pes(pes)
    
    Non zero centered.
    
    >>> ivn.plot_pes(pes, zero_centered= False)

    """
    
    pes=pes.groupby(level=[0,1,3]).mean() # Calculate means means across trials of the same event, per each signal, event and timestamp
    
    events = set(pes.index.levels[1])
    variables = set(pes.index.levels[0])
    
    combinations = sorted([*product(events,variables)]) # Each combinations of event and signal will be ploted in each axis
    # Calculate number of rows and columns
    nrows = len(events)
    ncols = len(variables)
    # Calculate the figure width and height
    fig_width = 3.2 * ncols + 1.5 + 0.2 # Width of 3.2 inches per subplot plus 1.5 inches of left edge plus 0.2 inches of right edge
    fig_height = 3.2 * ncols * (aspect * nrows / ncols) +1 + 0.2 # The Height depends on width, plus 1 inch of bottom edge plus 0.2 inches of right edge
    
    # Plot
    fig, axs=plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', figsize=(fig_width, fig_height)) # Plot each event-continuous variable combination in an axis
    plt.subplots_adjust(left=1.5/fig_width, right = (1 - 0.2/fig_width), bottom=1/fig_height, top = (1- 0.2/fig_height))
    #Function to make graph of peri-event spectogram
    def plot(ax, cell, data, zero_centered=zero_centered, aspect=aspect):
        df=data.loc[(cell[1], cell[0]), :].T.sort_index(ascending=False)
        if zero_centered:
            vmin, vmax= -(df.abs().max(axis=None)), (df.abs().max(axis=None))
            im=ax.imshow(df.values, cmap='seismic', vmin=vmin, vmax=vmax)
        else:
            im=ax.imshow(df.values, cmap='seismic')
        
        
        
        ax.set_xticks(np.linspace(0,df.shape[1],5), labels=np.linspace(df.columns[0], df.columns[-1], 5).round(2))
        ax.set_yticks(np.linspace(0,df.shape[0],5), labels=np.linspace(df.index[0], df.index[-1], 5).round())
        ax.set_title('{}, {}'.format(cell[0], cell[1]), fontweight='bold', y=1)
        ax.tick_params(axis='y')
        ax.tick_params(axis='x', labelrotation = -35)
        cbar = ax.figure.colorbar(im, ax=ax) # Create colorbar
        
        ax.set_aspect("auto", adjustable=None)
        ax.set_box_aspect(aspect)


    if len(combinations)<= 1:
        plot(axs, combinations[0], pes)
    
    else:
        for ax, cell in zip(axs.ravel(), combinations): # Loop over axes and combinations of events and signals and plot
            plot(ax, cell, pes)
    fig.supxlabel("Peri-event time")
    fig.supylabel("Frequency (Hz)")
    plt.show()        


    
class PeriEventSpectogram():
    """
    Create a PeriEventHistogram object.
    
    
    Parameters
    ----------
    data : pandas.DataFrame
        Multi-index pandas.DataFrame as returned by peri_event_spectogram() function, with original continuous variable names, event names, event trial number and peri-event time as index, frequencies as columns and power values as data.
    norm : str
        Normalization state of the data, it can be "None", "Condition_average", "Condition_specific" or "Trial_specific".
        
    
    Attributes
    ----------
    data: pandas.DataFrame
        Multi-index pandas.DataFrame with peri-event spectogram data, with original continuous variable names, event names, event trial number and peri-event time as index, frequencies as columns and power values as data.
    variable_names: list
        Names of each original continuous variable used to make the spectograms.
    event_names: list
        Names of each reference event.
    timestamps: list
        Timestamps of the peri-event spectogram.
    frequencies: list
        Frequencies included in the spectogram.
    normalization: normalization state of the data.
    
    Methods
    -------
    
    normalize():
        Normalize data using the formula (X - Mean(baseline)) / Mean(baseline) for each frequency, where X is every value and Mean(baseline) is the mean value of a baseline.
        
        Parameters
        ----------
        baseline : tuple, optional
            lowest and highest time limits of the baseline, or None. If None, baseline = (lowest_lim, highest_lim) of the argument passed to pes. The default is None.
        method : str, optional
            Method used to nromalize.
            "Condition_average": Mean(baseline) is calculated across all trials of all events and used to normalize the values of the peri-event spectogram.
            "Condition_specific": Mean(baseline) is calculated across all trials of each event, and used to normalize the data that specific event. This can be convenient when baselines differ between events.
            "Trial_specific": Mean(baseline) is calculated for each trial of each event, and used to normalize only that trial.
            The default is 'Condition_average'.
        inplace: bolean, optional
            If True, modifies the current object. If False, returns a new object with normalized data. The default is False.
        
        Raises
        ------
        TypeError
            If baseline is neither a tuple or None.
        NameError
            If method is neither "Condition_average", "Condition_specific" or "Trial_specific".
        ValueError
            If normalization attribute is not "None", which indicates that data is already normalized.
        
        
        Returns: PeriEventSpectogram
            If inplace is False, returns a PeriEventSpectogram object with normalized data, and normalization attribute as the method argument passed.
            
    slice_time(new_limits, inplace=False):
        
        Slice timestamps.
        
        Parameters
        ----------
        new_limits: tuple
            New lowest and highest limits of time.
        inplace: bolean, optional
            If True, modifies the current object. If False, returns a new object with sliced data. The default is False.
        
        Returns: PeriEventSpectogram
            New object with sliced timestamps
    
    slice_frequencies(new_limits, inplace=False):
        
        Slice frequencies.
        
        Parameters
        ----------
        new_limits: tuple
            Lowest and highest limits of the frequency.
        inplace: bolean, optional
            If True, modifies the current object. If False, returns a new object with sliced data. The default is False.
        
        Returns: PeriEventSpectogram
            New object with sliced frequecies.
    
    slice_events(event_list, inplace=False):
        
        Slice events.
        
        Parameters
        ----------
        event_list: list
            Event names to slice from the data.
        inplace: bolean, optional
            If True, modifies the current object. If False, returns a new object with sliced data. The default is False.
        
        Returns: PeriEventSpectogram
            New object with sliced events.
    
    calculate_means():
        
        Calculate means across trials of the same event for each variable, event name and timestamp.
        
    Returns: pandas.DataFrame
        Mean across trials of the same event of the peri-event spectogram. Multi-index pandas.DataFrame with original continuous variable names,
        event names and peri-event time as index, frequencies as columns and mean power values as data.
    
    plot(zero_centered=None, aspect=1, variables = None, evt_names = None):
        
        Plot peri-event spectograms, with each variable in a column and each event name in a row.

        Parameters
        ----------
        zero_centered : boolean or None, optional
            If True, colorscale is centered to zero. If None, colorscale is centered to zero if data is normalized (normalization attribute is either "Condition_average", "Condition_specific" or "Trial_specific"),
            but it is not centered to zero if data is not normalized (normalization attribute is "None"). The default is None.
            
        aspect : float, optional
            The y/x ratio of the axes aspect. The default is 1.
        
        variables: list or None, optional
            Subset of variables names to plot. If None, all variables are ploted. Default is None.
        
        evt_names: list or None, optional
            Subset of events to plot. If None, all events are ploted. Default is None.
        
        Returns
        -------
        None.
    
    """
    
    def __init__ (self, data, norm):
        
        if norm not in  ['None', 'Condition_average', 'Condition_specific', 'Trial_specific']:
            raise ValueError('norm must be either "None", "Condition_average", "Condition_specific" or "Trial_specific", but {} was found.'.format(norm))
        self.data = data
        self.variable_names= self._set_variable_names()
        self.event_names= self._set_event_names()
        self.timestamps= self._set_timestamps()
        self.frequencies = self._set_frequencies()
        self.normalization = norm
    
    def __str__(self):
        return '{} \n\nData normalization: {}'.format(self.data, self.normalization)
    
    def __repr__(self):
        return 'PeriEventSpectogram({},"{}")'.format(self.data, self.normalization)
    
    
    def _set_variable_names(self):
        return sorted([*set(self.data.index.levels[0])])
    
    def _set_event_names(self):
        return sorted([*set(self.data.index.levels[1])])
    
    def _set_timestamps(self):
        return sorted([*(set(self.data.index.levels[3]))])
    
    def _set_frequencies(self):
        return list(self.data.columns)

    
    def normalize (self, baseline=None, method='Condition_average', inplace=False):
        
        # Don't normalyze if data is already normalized
        if self.normalization != 'None':
            raise ValueError('Data is already normalized, no procesing was done!')
            
        if inplace:
            self.data = normalize_pes(self.data, baseline=baseline, method=method)
            self.normalization = method
        else:
            return PeriEventSpectogram(normalize_pes(self.data, baseline=baseline, method=method), norm=method)

     
    def slice_time (self, new_limits, inplace=False):
        new_data = self.data.loc[(slice(None),slice(None),slice(None),slice(new_limits[0],new_limits[1])),:]
        new_data.index = new_data.index.remove_unused_levels()
        if inplace:
            self.data = new_data
            self.timestamps = self._set_timestamps()
        else:
            return PeriEventSpectogram(new_data, norm=self.normalization)
    
    def slice_frequencies(self, new_limits, inplace=False):
        new_data = self.data.loc[:, slice(new_limits[0],new_limits[1])]
        if inplace:
            self.data = new_data
            self.frequencies = self._set_frequencies()
        else:
            return PeriEventSpectogram(new_data, norm=self.normalization)
    
    def slice_events(self, event_list, inplace=False):
        new_data  = self.data.loc[(slice(None),event_list,slice(None),slice(None)),:]
        new_data.index = new_data.index.remove_unused_levels()
        
        if inplace:
            self.data = new_data
            self.event_names= self._set_event_names()
        else:
            return PeriEventSpectogram(new_data, norm=self.normalization)
    
    def calculate_means(self):
        return self.data.groupby(level=[0,1,3]).mean()
    
    def plot(self, zero_centered=None, aspect=1, variables = None, evt_names = None):
        
        if zero_centered==None:
            if self.normalization == 'None':
                zero_centered = False
            else:
                zero_centered = True
        
        # Assign class attributes as defaults to method argument
        if variables is None:
            variables = self.variable_names
        
        if evt_names is None:
            evt_names = self.event_names
        
        df=self.data.loc[(variables, evt_names), :]
        plot_pes (df, zero_centered=zero_centered, aspect=aspect)

