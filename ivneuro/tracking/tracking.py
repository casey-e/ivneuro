# -*- coding: utf-8 -*-
"""
A module with functions for processing positions data based on xy coordinates.
"""

import numpy as np
import pandas as pd
            
  
def scale_centroids(centroids, x_dim,y_dim):
    """
    Scale centroids to match the real values in appropiate scale. It assumes position at lower and higher limits of each dimension have been recorded.

    Parameters
    ----------
    centroids : numpy.ndarray of shape (2 x n_timestamps)
        Values of X_centroid and Y_centroid to be scaled.
    x_dim : float
        Value from end to end of X coordinate.
    y_dim : float
        Value from end to end of Y coordinate.

    Returns
    -------
    numpy.ndarray with scaled values.

    """
    x_min, x_max=centroids[0].min(),centroids[0].max()
    y_min, y_max=centroids[1].min(),centroids[1].max()
    x_dist=x_max-x_min
    y_dist=y_max-y_min
      
    scaled_dif=(x_dist/y_dist - x_dim/y_dim)/(x_dim/y_dim)
    if abs(scaled_dif) > 0.1:
        print('\nWarning: the observed and provided x/y ratios have a difference of above 10%, consider switching the order of X and Y dimensions or cleaning centroids data and scale again')
    
    print('\nDifference between observed and provided x/y ratios:',100*scaled_dif, '%')
    
    x_multiplying_factor=x_dim/x_dist
    y_multiplying_factor=y_dim/y_dist
    min_values=np.array([x_min,y_min]).reshape(2,1)
    multiplying_factors=np.array([x_multiplying_factor,y_multiplying_factor]).reshape(2,1)
    result=(centroids-min_values)*multiplying_factors
    return(result)

def calculate_speed(X_values, Y_values, timestamps, smooth=0):
    '''
    Calculate scalar speed through the following steps:
    1. Calculate dtime, dX=PosX[T-deltaT]-posX[T] and dY=PosY[T-deltaT]-posY[T]
    2. Smooth dX and dY using a Gaussian filter. Note: by applying the Gaussian filter to dX and dY instead of the speed
    avoids short movements or random noie of the camera  to propagate to the speed calculus, leaving only changes in position caused 
    by displacement
    3. Calculate scalar speed: sqrt(dX**2+dY**2) / dT

    Parameters
    ----------
    X_values : NUMPY ARRAY
        Positions in X coordinate. Must have the same lenght than Y_values and timestamps
    Y_values :  NUMPY ARRAY
        Positions in Y coordinate. Must have the same lenght than X_values and timestamps
    timestamps : NUMPY ARRAY
        Timestamps for every position. Must have the same lenght than X_values and Y_values
    smooth : Float, optional
        Defines the window of time a 2 x smooth and the width of the  Gaussian kernel as 1 x smooth,
        for the Gaussian filter. The default is 0, in which case the Gaussian filter is not appplied.

    Returns
    -------
    speed: numpy.ndarray
        Values of scalar speed

    '''
    #Evaluate if X_values, Y_values and timestamps ahve the same lenght, raise error if not
    if len(X_values)!=len(Y_values) or len(X_values)!=len(timestamps):
        raise ValueError ('X_values, Y_values and timestamps must have the same lenght')
    #Evaluate if X_values, Y_values and timestamps are of type np.array, raise error if not
    if type(X_values)!=np.ndarray or type(Y_values)!=np.ndarray or type(timestamps)!=np.ndarray:
        raise TypeError ('X_values, Y_values and timestamps must be of type numpy.ndarray')
    #Evaluate if smooth is of type int or float, raise error if not
    if type(smooth)!= float and type(smooth)!= int:
        raise TypeError('smooth must be integer or float')
    
    #Calculate nstantaneus change in time
    # dif_time=np.round(np.array((scaled_centroids.index-np.roll(scaled_centroids.index, 1))),3)
    dT=np.round(np.roll(timestamps, -1)-timestamps,3)
    dT[-1]=np.nan
    #Calculate average dif time, which will be used for the smoothing process
    avg_dT=np.mean(dT[0:-1])
    #Calculate the number of rows to use a std in the gaussian filter, by dividing the smooth time provided by the average delta time
    smooth2=smooth/avg_dT
    #Calculate the differential X and Y, by substracting each value to the next value (using shift)   
    dX=np.roll(X_values,-1)-X_values
    dX[-1]=np.nan #last value is wrong, because np.roll put the last value at firt position
    dY=np.roll(Y_values,-1)-Y_values
    dY[-1]=np.nan #last value is wrong, because np.roll put the last value at firt position
    if smooth2!=0:  # Only apply smoothing if smooth2 is different from 0  
        # if smooth2 is lower than 1, the use smooth2=1; otherwise, round it and transform it to integer
        if smooth2 <1:
            smooth2=1
        else: 
            smooth2=int(np.round(smooth2))
        #Apply gaussian filter to dX and dY
        dX=pd.Series(dX).rolling(2*smooth2, center=True, win_type='gaussian').mean(std=smooth2).round(3)
        dY=pd.Series(dY).rolling(2*smooth2, center=True, win_type='gaussian').mean(std=smooth2).round(3)
        dX=dX.values
        dY=dY.values
    #Use Pitagoras to calculate the differential position (cat2+cat2=hyp2)
    dPos=np.sqrt(dX**2+dY**2) 
    #Calculate speed, dividing dPos by dT
    speed=dPos/dT
    return(speed)
    

def position_of_event(x_values, y_values, timestamps, events, estimator=np.median):
    """
    Estimate the most likelly position at with an event occurs, or get all the positions where that event occurs.

    Parameters
    ----------
    x_values : np.ndarray of shape (1 X n_timestamps)
        All positions in x axis. Must be of the same lenght than y_values and timestamps
    y_values : np.ndarray of shape (1 X n_timestamps)
       All positions in y axis. Must be of the same lenght than x_values and timestamps
    timestamps : np.ndarray of shape (1 X n_timestamps)
        All timestamps. Must be of the same lenght than x_values and y_values.
    events : one dimensional numpy.array or list of floats
        Timestamps of the event.
    estimator : function, optional
        Function to be used to estimate the most likely position of the event. The default is np.median.
    

    Raises
    ------
    TypeError
        If x_values, y_values or timestamps is not np.ndarray.
    ValueError
        If x_values lenght, y_values lenght and timestamps lenght are not the same.

    Returns
    -------
    Tuple 
        Most likely position (x, y) if estimator is not None, or all the positions of the event (np.array([x])), np.array([y]) if estimator is None.

    """
   
    if len(x_values)!=len(y_values) or len(x_values)!=len(timestamps):
        raise ValueError ('x_values, y_values and timestamps must have the same lenght')
     #Evaluate if x_values, y_values and timestamps are of type np.array, raise error if not
    if type(x_values)!=np.ndarray or type(y_values)!=np.ndarray or type(timestamps)!=np.ndarray:
        raise TypeError ('x_values, y_values and timestamps must be of type numpy.ndarray')
    
    indexes=np.searchsorted(timestamps, events)
    positions_x=x_values[indexes]
    positions_y=y_values[indexes]
    
    if estimator == None:
        return (positions_x, positions_y)
    else:
        return (estimator(positions_x), estimator(positions_y))



def distances_to_position(x_values, y_values, position):
    """
    Given a set of x and y positions, calculate the distance to a specific position at each row.

    Parameters
    ----------
    x_values : np.ndarray of shape (1 X n_timestamps)
        X_centroid positions at each timestamp. Must be of the same lenght than y_values and timestamps
    y_values : np.ndarray of shape (1 X n_timestamps)
    position : tuple
        Position (x, y) to calculate distance.

    Returns
    -------
    np.ndarray
        One dimensional array with distances to position at every point.

    """
    
    return np.sqrt((x_values-position[0])**2+(y_values-position[1])**2)

