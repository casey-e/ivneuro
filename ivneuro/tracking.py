# -*- coding: utf-8 -*-
"""
A module with functions for processing positions data based on xy coordinates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
            
  
def scale_centroids(centroids, x_dim,y_dim):
    """
    Scale centroids to match the real values in appropiate scale.
    
    This function assumes that positions at lower and higher limits of each dimension have been recorded.

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
    
    Examples
    --------
    
    Use scale_centroids function.
    
    >>> # Create and array with x and y coordinates: centroids
    >>> import ivneuro as ivn 
    >>> import numpy as np
    >>> x = 10 * np.cos(np.linspace(0, 2*np.pi, 100)) + 10
    >>> y = 20 * np.sin(np.linspace(0, 2*np.pi, 100)) + 20 
    >>> centroids = np.concatenate((x.reshape(100,1), y.reshape(100,1)), axis=1).T # Coordinates for an oval trajectory
    >>> # Scale using scale_centroids function
    >>> scaled_centroids = ivn.scale_centroids(centroids, 15, 30)
    Difference between observed and provided x/y ratios: -0.012587232612482069 %
    >>> print(scaled_centroids[:,:5]) # Print first 5 values of x and y
    array([[15.        , 14.98489627, 14.9396459 , 14.8644311 , 14.75955473],
           [15.        , 15.95147856, 16.89912585, 17.83912603, 18.76769406]])
    
    If ratios between observed and provided x and y values are too different, scale_centroids will through a warning message.

    >>> # Scale with inverted x and y measures
    >>> scaled_centroids = ivn.scale_centroids(centroids, 30, 15)
    Warning: the observed and provided x/y ratios have a difference of above 10%, consider switching the order of X and Y dimensions or cleaning centroids data and scale again
    Difference between observed and provided x/y ratios: -75.00314680815312 %

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
    Calculate scalar speed.
    
    Speed is calculated through the following steps:
    1. Calculate dtime, dX=PosX[T-deltaT]-posX[T] and dY=PosY[T-deltaT]-posY[T].
    2. Smooth dX and dY using a Gaussian filter. Note: by applying the Gaussian filter to dX and dY instead of the speed
    avoids short movements or random noie of the camera  to propagate to the speed calculus, leaving only changes in position caused 
    by displacement.
    3. Calculate scalar speed: sqrt(dX**2+dY**2) / dT.

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
    
    Examples
    --------
    
    Create x coordinates, y coordinates and  timestamps. Calculate speed and print it.
    
    >>> import ivneuro as ivn 
    >>> import numpy as np
    >>> x = 10 * np.cos(np.linspace(0, 2*np.pi, 100)) + 10
    >>> y = 20 * np.sin(np.linspace(0, 2*np.pi, 100)) + 20 
    >>> ts = np.linspace(0,20,100).round(1)
    # Calculate speed with calculate_speed function and print the first 5 values
    >>> speed=ivn.calculate_speed(x, y, ts)
    >>> print(speed[:5])
    [6.3431908  6.32404896 6.28590074 6.22901826 6.15381269]

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
    Estimate the most likely position at with an event occurs, or get all the positions where that event occurs.

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
    
    Examples
    --------
    
    Estimate the position of an event with the median.
    
    >>> # Create x coordinates, y coordinates, centroids timestamps and event.
    >>> import ivneuro as ivn 
    >>> import numpy as np
    >>> x = 10 * np.cos(np.linspace(0, 2*np.pi, 100)) + 10
    >>> y = 20 * np.sin(np.linspace(0, 2*np.pi, 100)) + 20 
    >>> ts = np.linspace(0,20,100).round(1)
    >>> np.random.seed(24)
    >>> event = np.random.choice(ts[(ts>10) & (ts<15)], size=5, replace = False) # Create an event by making a random choice from timestamps  
    >>> # Estimate the position with position_of_event, return the median
    >>> ivn.position_of_event(x, y, ts, event)
    (1.7632341857016716, 8.658802722744587)
    
    Return the mean of the positions
    
    >>> ivn.position_of_event(x, y, ts, event, estimator = np.mean)
    (3.356415004157808, 7.336570446958302)
    
    Return all the positions for x and y
    
    >>> ivn.position_of_event(x, y, ts, event, estimator = None)
    (array([5.        , 0.83891543, 8.57685162, 0.60307379, 1.76323419]),
     array([ 2.67949192, 11.98138929,  0.20357116, 13.15959713,  8.65880272]))
    

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
    
    Examples
    --------
    
    Calculate distances to a position.
    
    >>> # Create x coordinates and y coordinates, create position of interest
    >>> import ivneuro as ivn 
    >>> import numpy as np
    >>> x = 10 * np.cos(np.linspace(0, 2*np.pi, 100)) + 10
    >>> y = 20 * np.sin(np.linspace(0, 2*np.pi, 100)) + 20
    >>> center = (10, 20) # Position of interest
    >>> # Calculate distances
    >>> distances = ivn.distances_to_position (x, y, center)
    >>> print(distances[:5])
    [10.         10.06015795 10.23756293 10.523536   10.90516361]

    """
    
    return np.sqrt((x_values-position[0])**2+(y_values-position[1])**2)



class EventPosition():
    
    """
    Obtain the position of an event.
    
    
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
    ValueError
        If estimator == None.
        
    
    Attributes
    ----------
    x_values : np.ndarray
        All positions in x axis.
    y_values : np.ndarray
        All positions in y axis.
    timestamps: np.ndarray
        All timestamps
    events: np.ndarray
        Timestamps of the event.
    estimator: function
        The function used to estimate the most likelly position.
    estimated_position: tuple 
        Most likely position (x, y).
    position_std: tuple
        Standar deviation for the positions of th event (std(x), std(y))
    
    
    Methods
    -------
    distances_to_event()
        Calculate the distance to the event estimated position at each timestamp
    
        Returns: np.ndarray
            One dimensional array with distances.
    
    set_distances_to_event()
        Set distances_to_event attribute using distances_to_event function
        
        Returns:
            None
    
    all_event_positions():
        Get all the positions at wich the event occurrs.
        
        Returns: tuple
            All the positions of the event (np.array([x])), np.array([y]).
    
    
    get_event_position_std():
        Calculate the standar deviation for the event position at x and y axis.
        
        Returns: tuple
            Standar deviation in x and y (std(x), std(y))
    
    plot():
        plot the all the trajectory, positions where the event occurred and the event estimated position
        
        Returns: 
            None
    
    
    """
    
    def __init__(self, x_values, y_values, timestamps, events, estimator=np.median):
         self.x_values = x_values
         self.y_values = y_values
         self.timestamps = timestamps
         self.events = events
         self.estimator=estimator
         self.estimated_position = self._position_of_event()
         self.position_std=self.get_event_position_std()
    
    def __str__(self):
        return 'Estimated position: ({} ± {}, {} ± {}), (estimator ± std)'.format(self.estimated_position[0],  self.position_std[0], self.estimated_position[1],  self.position_std[1])
    
    
    def __repr__(self):
        return 'EventPosition(x_values={}, y_values = {}, timestamps= {}, events = {}, estimator= {})'.format(self.x_values,self.y_values,self.timestamps,self.events,self.estimator)
    
    def _position_of_event(self):
        if self.estimator == None:
            raise ValueError ('Estimator cannot be None')
        return position_of_event(self.x_values, self.y_values, self.timestamps, self.events, self.estimator)
         
         
    def distances_to_event(self):
        return distances_to_position(self.x_values, self.y_values, self.estimated_position)
         
    def set_distances_to_event(self):
        self.distances_to_event = distances_to_event(self)
        
    def all_event_positions(self):
        return position_of_event(self.x_values, self.y_values, self.timestamps, self.events, estimator = None)
    
    def get_event_position_std(self):
        x_val, y_val= position_of_event(self.x_values, self.y_values, self.timestamps, self.events, estimator = None)
        return (x_val.std(), y_val.std())
            
            
    def plot(self):
            
        # Create positions for all events
        all_evt_pos = self.all_event_positions()
            
        # Plot
        plt.plot(self.x_values,self.y_values,color='gray',alpha=0.5, label='Trajectory')
        plt.scatter(all_evt_pos[0],all_evt_pos[1], color='black',s=10, label='Event positions')
        plt.scatter(self.estimated_position[0],self.estimated_position[1], color='red', marker='D',s=50, label='Event estimated position')
        plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        plt.axis("equal")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title('Event positions')
        plt.show()