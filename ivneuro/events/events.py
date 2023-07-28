# -*- coding: utf-8 -*-
"""

User-facing functions for processing events.

"""

import numpy as np




def make_intervals(timestamps, start, end):
    """
    Make intervals based on timestamps

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

    """
    
    return [slice(ts+start,ts+end) for ts in timestamps]


def classify_events_base_on_time(event1,event2,treshold,mode='left'):
    """
    Clasiffy an event in two categories based on how close in time it occurs from an event of reference.

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
        only looks event1 that ocurr after event2; "both", look temporal proximity before and after. The default is 'left'.

    Returns
    -------
    near : np.array of shape (1 x o)
        Subset of event1 classified as temporally close to event2.
    far : np.array of shape (1 x p).
        Subset of event1 classified as temporally far from event2.

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