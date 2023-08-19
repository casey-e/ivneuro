# -*- coding: utf-8 -*-
"""

User-facing functions for processing events.

"""

import numpy as np




def make_intervals(timestamps, start, end):
    """
    Make intervals based on timestamps.

    Parameters
    ----------
    timestamps : np.ndarray or list
        Timestamps to use as reference to make intervals.
    start : int or float
        Start of interval, relative to timestamps.
    end : int or float
        End of interval, relative to timestamps.

    Returns
    -------
    list
        Slices with the start time and end time of each interval.
    
    Examples
    --------
    
    Create intervals using make_intervals function.
    
    >>> import ivneuro as ivn
    >>> event = [*range(10,40, 10)] # Timestamps
    >>> # Make intervals
    >>> intervals = ivn.make_intervals(event, 0, 3)
    >>> print(intervals)
    [slice(10, 13, None), slice(20, 23, None), slice(30, 33, None)]
    
    Use intervals to slice a pandas.DataFrame
    
    >>> # Create dataframe
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(24)
    >>> df=pd.DataFrame(data=np.random.rand(40), columns=['values'])
    >>> # Slice datafrane
    >>> [df.loc[i,] for i in intervals]
    [      values
     10  0.320519
     11  0.366415
     12  0.709652
     13  0.900142,
           values
     20  0.842780
     21  0.306013
     22  0.631170
     23  0.680239,
           values
     30  0.486032
     31  0.807219
     32  0.844220
     33  0.534681]
    

    """
    
    return [slice(ts+start,ts+end) for ts in timestamps]


def classify_events_base_on_time(event1,event2,treshold,mode='left'):
    """
    Classify an event in two categories based on how close in time it occurs from an event of reference.

    Parameters
    ----------
    event1 : numpy.array of shape (1 x n)
        Event to classify.
    event2 : numpy.array of shape (1 x m)
        Event of reference.
    treshold : TYPE
        Threshold amount of time used to classify events.
    mode : str, optional
        Define the mode of evaluation of proximity. "left", only looks event1 that occur before event2; "right", 
        only looks event1 that ocurr after event2; "two-sides", look temporal proximity before and after. The default is 'left'.

    Returns
    -------
    near : np.array of shape (1 x o)
        Subset of event1 classified as temporally close to event2.
    far : np.array of shape (1 x p).
        Subset of event1 classified as temporally far from event2.
    
    Examples
    --------
    
    Classify events with classify_events_base_on_time function.
    
    >>> # Import packages and create events
    >>> import ivneuro as ivn
    >>> import numpy as np
    >>> event1 = np.array([3, 7, 10.5, 15.3])
    >>> event2 = np.array([1, 7.5, 15])
    >>> # Classify events
    >>> near, far = ivn.classify_events_base_on_time(event1, event2, treshold = 1)
    >>> print(near)
    [7.]
    >>> print(far)
    [ 3.  10.5 15.3]
    
    Set mode to 'right'.
    
    >>> near, far = ivn.classify_events_base_on_time(event1, event2, treshold = 1, mode = 'right')
    >>> print(near)
    [15.3]
    >>> print(far)
    [ 3.   7.  10.5]
    
    Set mode to 'both'.
    
    >>> near, far = ivn.classify_events_base_on_time(event1, event2, treshold = 1, mode = 'two-sides')
    >>> print(near)
    [ 7.  15.3]
    >>> print(far)
    [ 3.  10.5]
    

    """
    near=[]
    far=[]
    for i in event1:
        j=event2-i
        if mode=='left':
            if len(j[j>0])>0 and np.min(j[j>0])<=treshold:
                near.append(i)
            else:
                far.append(i)
        
        elif mode == 'two-sides':
            if np.min(abs(j))<=treshold:
                near.append(i)
            else:
                far.append(i)
        
        else:
            if len(j[j<0])>0 and abs(np.max(j[j<0]))<=treshold:
                near.append(i)
            else:
                far.append(i)
    near=np.array(near)
    far=np.array(far)
    return near, far