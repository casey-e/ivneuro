# -*- coding: utf-8 -*-
"""
A module for PeriEventHistogram class.

Peri-event histograms analysis can be higly demanding, depending on the size and amount of continuous 
variables and the amount of reference events and number of trials of each of event. PeriEventHistogram
class aims to facilitate processing, interpretation and visualization of peri-event histograms without
running the analysis again. It is a pandas DataFrame Subclass and is returned by default by
ivneuro.continuous.peh function.

"""

from itertools import product
import pandas as pd
import matplotlib.pyplot as plt


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

