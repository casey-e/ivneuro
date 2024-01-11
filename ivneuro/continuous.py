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
from scipy.ndimage import label, find_objects


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
    
    Examples
    --------
    
    Calculate sample period of a timeserie.
    
    >>> # Create timeserie
    >>> import ivneuro as ivn
    >>> import numpy as np
    >>> timestamps = np.arange(0, 100, 0.001) # Timeserie of period 0.001
    >>> # Use the function to calculate sample period.
    >>> ivn.calculate_sampling_period(timestamps)
    0.001000000000000334
    
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
    
    Examples
    --------
    
    Calculate sample rate of a timeserie.
    
    >>> # Create timeserie
    >>> import ivneuro as ivn
    >>> import numpy as np
    >>> timestamps = np.arange(0, 100, 0.001) # Timeserie of period 0.001
    >>> # Use the function to calculate sample rate.
    >>> ivn.calculate_sampling_rate(timestamps)
    1000.0

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
    
    Examples
    --------
    
    Create event and signals.
    
    >>> import ivneuro as ivn
    >>> import pandas as pd
    >>> # Create event: burst
    >>> burst = [*range(30,300, 30)]
    >>> # Generate signals
    >>> signal1 = ivn.generate_signal(300, burst, 2, burst_duration = 0.5, burst_amplitude=1) 
    >>> signal2 = ivn.generate_signal(300, burst, 0.5, burst_duration = 2, burst_amplitude=1.5, seed = 21) 
    >>> signal3 = ivn.generate_signal(300, burst, 0.2, burst_duration = 3, burst_amplitude=1.5, seed = 10)
    >>> signals = pd.concat([signal1, signal2, signal3], axis = 1)  
    >>> # Use peh_list to create a list of peri-event histograms
    >>> hist_list=ivn.continuous.peh_list(signals, burst, lower_lim = -5, higher_lim = 5)
    >>> hist_list[0] # Print first element
              Signal 2Hz  Signal 0.5Hz  Signal 0.2Hz
    1 -5.000    0.308629      0.802330     -0.772324
      -4.999    0.383986      0.841905     -0.842523
      -4.998    0.337540      0.966326     -0.901157
      -4.997    0.301379      0.992252     -0.832871
      -4.996    0.396404      1.047770     -0.900500
                 ...           ...           ...
       4.996    0.211593     -2.017009     -0.826280
       4.997    0.324347     -2.130981     -0.891831
       4.998    0.229633     -2.077634     -0.974507
       4.999    0.216934     -2.014184     -0.940523
       5.000    0.157096     -2.007946     -1.020650
    [10001 rows x 3 columns]

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
    
    Examples
    --------
    
    Create event and signals.
    
    >>> import ivneuro as ivn
    >>> import pandas as pd
    >>> # Create events: burst and control
    >>> burst = [*range(30,300, 30)]
    >>> control = [*range(45,300, 30)]
    >>> events = {'burst':burst,'control':control}
    >>> # Generate signals
    >>> signal1 = ivn.generate_signal(300, burst, 2, burst_duration = 0.5, burst_amplitude=1) 
    >>> signal2 = ivn.generate_signal(300, burst, 0.5, burst_duration = 2, burst_amplitude=1.5, seed = 21) 
    >>> signal3 = ivn.generate_signal(300, burst, 0.2, burst_duration = 3, burst_amplitude=1.5, seed = 10)
    >>> signals = pd.concat([signal1, signal2, signal3], axis = 1)
    
    Peri-event histograms for a single variable and a single event (with multiple trials).
    
    >>> hist = ivn.peh(signal1, burst, lower_lim = -5, higher_lim = 5) # histograms from 5 seconds before to 5 seconds after each trial
    >>> type(hist)
    ivneuro.continuous.PeriEventHistogram
    >>> hist
                                    Signal 2Hz
    Event_name Event_number Time              
    Event      1            -5.000    0.308629
                            -4.999    0.383986
                            -4.998    0.337540
                            -4.997    0.301379
                            -4.996    0.396404
                                       ...
               9             4.996    1.773434
                             4.997    1.768014
                             4.998    1.742938
                             4.999    1.684667
                             5.000    1.573737
    [90009 rows x 1 columns]
    
    
    Return a pandas.DataFrame instead of a ivneuro.continuous.PeriEventHistogram.
    
    >>> hist = ivn.peh(signal1, burst, lower_lim = -5, higher_lim = 5, return_DataFrame = True) # histograms from 5 seconds before to 5 seconds after each trial
    >>> type(hist)
    pandas.core.frame.DataFrame
    
    Peri-event histogram for multiple variables and multiple events (with multiple trials each).
    
    >>> hist = ivn.peh(signals, events, lower_lim = -3, higher_lim = 3) # histograms from 3 seconds before to 3 seconds after each trial
    >>> hist.variable_names
    ['Signal 2Hz', 'Signal 0.5Hz', 'Signal 0.2Hz']
    >>> hist.event_names
    ['control', 'burst']
    >>> hist.calculate_means()
                       Signal 2Hz  Signal 0.5Hz  Signal 0.2Hz
    Event_name Time                                          
    burst      -3.000   -0.323144      0.421450      0.436888
               -2.999   -0.234888      0.370375      0.440265
               -2.998   -0.232368      0.398991      0.406327
               -2.997   -0.233245      0.368178      0.381116
               -2.996   -0.223524      0.319660      0.367928
                          ...           ...           ...
    control     2.996    0.245362      0.387958      0.362177
                2.997    0.270217      0.431107      0.326198
                2.998    0.263590      0.420351      0.303861
                2.999    0.335115      0.444825      0.323329
                3.000    0.361686      0.388019      0.330680
    [12002 rows x 3 columns]
    

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



def make_itervals_from_continuous (contvar, timestamps, higher_than=None, lower_than=None, min_duration=0):
    """
    Make intervals of time when a continuous variable have values higher or lower than a treshols, or between 2 values.

    Parameters
    ----------
    contvar : one-dimensional numpy.ndarray
        Continuous variable values.
    timestamps : one-dimensional numpy.ndarray
        Continuous variable timestamps.
    higher_than : int or float, optional
        Lowher treshold. The default is None.
    lower_than : int or float, optional
        Higher treshold. The default is None.
    min_duration : float, optional
        Minimum duration the condition must be met for a slice of time to be included in the resulting interval. The default is 0.

    Raises
    ------
    TypeError
        If higher_than or lower_than is not int or float.
    ValueError
        If both higher_than and higher_than are not None, and higher_than is not lower_than lower_than.

    Returns
    -------
    slices : list if slices
        List of timestamps slices corresponding to intervals that met the criteria.
    
    
    
    Examples
    --------
    
    >>> import ivneuro as ivn
    >>> import numpy as np
    >>> np.random.seed(46)
    # Make timestamps
    >>> timestamps=np.linspace(0,2.5, 26)
    # Make the continuous values as a quadratic function of timestamps, centered at x=1
    >>> contvar=(timestamps-1)**2

    Intervals for time windows with contvar >= 0.5
    
    >>> make_itervals_from_continuous (contvar, timestamps, higher_than=0.5)
    [slice(0.0, 0.2, None), slice(1.8, 2.5, None)]

    Intervals for time windows with contvar >= 0.5 lasting at least 0.4 seconds
    
    >>> make_itervals_from_continuous (contvar, timestamps, higher_than=0.5, min_duration=0.4)
    [slice(1.8, 2.5, None)]

    Intervals for time windows with contvar <= 0.5
    
    >>> make_itervals_from_continuous (contvar, timestamps, lower_than=0.5)
    [slice(0.30000000000000004, 1.7000000000000002, None)]

    Intervals for time windows with contvar values between 0.4 and 0.8
    
    >>> make_itervals_from_continuous (contvar, timestamps, higher_than=0.4, lower_than=0.8)
    [slice(0.2, 0.30000000000000004, None), slice(1.7000000000000002, 1.8, None)]
    

    """
    
    # Raise a TypeError if higher_than or lower_than are not float or int
    if not isinstance(higher_than, (int, float)) and not isinstance(lower_than, (int, float)):
        raise TypeError('higher_than or lower_than, or both, must be either int or float')
    # If only higher_than is numeric, use higher_than to transform contvar in binary
    if not isinstance(lower_than, (int, float)):
        mask = np.where(contvar >= higher_than, 1, 0)
    # If only lower_than is numeric, use lower_than to transform contvar in binary
    elif not isinstance(higher_than, (int, float)):
        mask = np.where(contvar <= lower_than, 1, 0)
    
    # If higher_than and lower_than are numeric, use them to transform contvar in binary
    else:
        # Raise a ValueError if higher_than is not higher than lower_than
        if higher_than >= lower_than:
            raise ValueError('higher_than must be lower than lower_than')
        mask = np.where((contvar >= higher_than) & (contvar <= lower_than), 1, 0)
    
    sr=ivn.calculate_sampling_rate(timestamps)
    min_size=int(min_duration*sr) # Calculate the minimum size to filter slices
    
    # Generate structuring element
    s = [1,1,1] 
    # Label every interval where conditions are met whithout interruptions
    labelled = label(mask, s)[0] 
    # Make slices for every period
    slices=find_objects(labelled)
    # Filter for slices larger than min_size
    slices=[i[0] for i in slices if labelled[i].size>min_size] 
    # Transform index values to timestamps values
    slices = [slice(j[0], j[-1]) for j in [timestamps[i] for i in slices]] 
    
    return slices