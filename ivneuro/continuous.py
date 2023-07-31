# -*- coding: utf-8 -*-
"""

User-facing functions for analysis of continuous variables, and PeriEventHistogram class.

Peri-event histograms analysis can be higly demanding, depending on the size and amount of continuous 
variables and the amount of reference events and number of trials of each of event. PeriEventHistogram
class aims to facilitate processing, interpretation and visualization of peri-event histograms without
running the analysis again. It is a pandas DataFrame Subclass and is returned by default by
ivneuro.continuous.peh function.

"""

import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from .utils import significant_decimal_positions


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
       
    # Use contvar.iloc[np.searchsorted(contvar.index, evt),].index to get the list of indexes of contvar dataframe, and 
    # contvar.loc[(i+lower_lim):(i+higher_lim)] to slice contvar dataframe around each event timestamp
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
    

class PeriEventHistogram(pd.DataFrame):
    
    """
    Create a PeriEventHistogram object.
    
    PeriEventHistogram class inherits from pandas.DataFrame and adds functionalities for easily extract information from the data and plot it.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Multi-index pandas.DataFrame as returned by peh() function, with event names, event trial number and peri-event time as index, continuous variables as columns and values as data.
        
    
    Attributes
    ----------
    
    variable_names: list
        Names of each continuous variable.
    event_names: list
        Names of each reference event.
    timestamps: list
        Timestamps of the peri-event histogram.

    Methods
    -------
       
    slice_time(new_limits):
        
        Slice timestamps.
        
        Parameters
        ----------
        new_limits: tuple
            New lowest and highest limits of time.
        
        Returns: PeriEventHistogram
            New object with sliced timestamps
    
    
    slice_events(event_list):
        
        Slice events.
        
        Parameters
        ----------
        event_list: list
            Event names to slice from the data.
        
        Returns: PeriEventHistogram
            New object with sliced events.
    
    calculate_means():
        
        Calculate means across trials of the same event for each variable, event name and timestamp.
        
    Returns: pandas.DataFrame
        Mean across trials of the same event of the peri-event histograms. Multi-index pandas.DataFrame with event names and peri-event time as index, 
        continuous variable names as columns and mean variable values as data.
    
    plot(aspect=1, cont_names = None, evt_names = None, sharey='all'):
        
        Plot peri-event histograms, with each variable in a column and each event name in a row.

        Parameters
        ----------
        aspect : float, optional
            The y/x ratio of the axes aspect. The default is 1.
        
        cont_names: list or None, optional
            Subset of continuous variables names to plot. If None, all variables are ploted. Default is None.
        
        evt_names: list or None, optional
            Subset of events to plot. If None, all events are ploted. Default is None.
        
        sharey: bool or {'none', 'all', 'row', 'col'}, optional
            Parameter of matplotlib.pyplot.subplots() to control sharing of properties among y axis. Refer to matplotlib.pyplot.subplots in matplotlib manual for more information.
            True or 'all': x- or y-axis will be shared among all subplots.
            False or 'none': each subplot x- or y-axis will be independent.
            'row': each subplot row will share an x- or y-axis.
            'col': each subplot column will share an x- or y-axis.
            The default is 'all'.
        
        Returns
        -------
        None.
    
    """
    
    def __init__(self, *args,**kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.variable_names= self._set_variable_names()
        self.event_names= self._set_event_names()
        self.timestamps= self._set_timestamps()

    
    
    def _set_variable_names(self):
        return list(self.columns)
    
    def _set_event_names(self):
        return [*set(self.index.levels[0])]
    
    def _set_timestamps(self):
        return sorted([*(set(self.index.levels[2]))])
    
    def slice_time (self, new_limits):       
        new_data = self.loc[(slice(None),slice(None),slice(new_limits[0],new_limits[1])),:]
        new_data.index = new_data.index.remove_unused_levels()
        return PeriEventHistogram(new_data)
    
    def slice_events(self, event_list):
        new_data  = self.loc[event_list,:]
        new_data.index = new_data.index.remove_unused_levels()
        return PeriEventHistogram(new_data)
    
    def calculate_means(self):
        return self.groupby(level=[0,2]).mean()
    
    
    def plot(self, aspect=1, cont_names = None, evt_names = None, sharey='all'):
        
        
        # Assign class attributes as defaults to method argument
        if cont_names is None:
            cont_names = self.variable_names
        
        if evt_names is None:
            evt_names = self.event_names

        combinations = sorted([*product(set(evt_names),set(cont_names))]) # Combinations of event (index) and continuous variables (columns) to iterate when plotting
        
        df=self[cont_names] # Slice columns of continuous values of interest
        df=df.groupby(level=[0,2]).mean() #Calculate averages per event and timestamp
        
        
        # Calculate number of rows and columns
        nrows = len(evt_names)
        ncols = len(cont_names)
        
        # Calculate the figure width and height
        fig_width = 3.2 * ncols  + 0.2 + 0.2 # Width of 3.2 inches per subplot plus 0.2 inches of left edge plus 0.2 inches of right edge
        fig_height = 3.2 * ncols * (aspect * nrows / ncols) +1 + 0.2 # The Height depends on width, plus 1 inch of bottom edge plus 0.2 inches of right edge
        
        # Plot
        fig, axs=plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey=sharey, figsize=(fig_width, fig_height)) # Plot each event-continuous variable combination in an axis
        plt.subplots_adjust(left=0.2/fig_width, right = (1 - 0.2/fig_width), bottom=1/fig_height, top = (1- 0.2/fig_height))
        
        #Function to make graph of peri-event histogram
        def plot(ax, cell, data, aspect = aspect):
                current_df=data.loc[(cell[0],), cell[1]]
                current_df.plot(ax=ax)
                ax.set_title('{}, {}'.format(cell[0], cell[1]), fontweight='bold', y=0.98)
                # ax.set_xlabel("Peri-event time")
                ax.tick_params(axis='y', labelrotation = 45)
                ax.axvline(x=0, ls= '--', c='gray')
            
                ax.set_aspect("auto", adjustable=None)
                ax.set_box_aspect(aspect)
        
        if len(combinations)<= 1:
            plot(axs, combinations[0], df)
        
        else:
            for ax, cell in zip(axs.ravel(), combinations): # Loop over axes and combinations of events and signals and plot
                plot(ax, cell, df)
        fig.supxlabel("Peri-event time")
        plt.show()