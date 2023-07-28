# -*- coding: utf-8 -*-
"""
A module with EventPositions class, wich allows to calculate and plot the positions of ocurrence
of an event.
"""
import numpy as np
import matplotlib.pyplot as plt
from ivneuro.tracking.tracking import position_of_event, distances_to_position

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
        plt.legend(framealpha=0.5)
        plt.axis("equal")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.title('Event positions')
        plt.show()
        
