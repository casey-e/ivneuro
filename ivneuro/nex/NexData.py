# -*- coding: utf-8 -*-
"""
A module with NexData class, wich puts together all the functions of pull_data subpackage and creates a
NexData object, wich contains the extrated variables as attributes.
"""

from .pull_nex_data import pull_fp, pull_events, pull_continuous, pull_neurons, pull_markers, pull_centroids, reader


class NexData():
    """
    Extract variables from a .Nex5 file.
    
     
    Parameters
    ----------
    
    file : str
        Complete path of the .Nex5 file to extract data from.
        
    FP_of_interest : list, optional
        List of names of field potentials to extract. If None, nothing is extracted; if an empty list, all the field potentials are extracted. The default is [].
    
    events_of_interest : list, optional
        List of names of events to extract. If None, nothing is extracted; if an empty list, all the events are extracted. The default is [].

    continuous_of_interest : list, optional
        List of names of continuous to extract. If None, nothing is extracted; if an empty list, all non-field potential continuous variables are extracted. The default is [].

    neurons_of_interest : list, optional
        List of names of events to extract. If None, nothing is extracted; if an empty list, all the neurons are extracted. The default is [].
     
    markers_of_interest : list, optional
        List of names of markers to extract. If None, nothing is extracted; if an empty list, all the markers are extracted. The default is [].
    
    centroids_of_interest : list, optional
        List of names of centroids to extract. If None, nothing is extracted; if an empty list, all centroid variables are extracted. The default is [].
    
    clear_Nex_data : bool, optional
        If True, data atribute is set to None. Otherwise, result of reader.ReadNex5File(file) is stored in data atribute. The default is True.
    
    
    Attributes
    ----------
    file_path : str
        File path of the .Nex5 file the data was extracted from.
    
    FP : pandas.DataFrame
        Field potentials (in mV) in each column and timestaps as index.
    
    FP_sampling_rate : float
        Sampling rate of local field potentials.
    
    events : dict
        Event names as keys and list of timestamps as values.
    
    continuous : list of pandas.DataFrames
        Each DataFrame of the list contains a column with the continuous values and the timestamps as index,for every continuous variable (excluding field potentials).
    
    neurons : dict
        Neuron names as keys and list of timestamps as values.
    
    markers : dict
        Markers names as keys and list of timestamps as values.
    
    centroids : pandas.DataFrame
        Centroids in each column and timestaps as index.
    
    
    Methods
    -------
    clear_fileData()
        Set data atribute to None.
        
    """
    

    def __init__(self, file, FP_of_interest=[],  events_of_interest=[], continuous_of_interest=[], neurons_of_interest=[], markers_of_interest=[], centroids_of_interest=[], clear_Nex_data=True):
        self.file_path = file
        self._FP_of_interest=FP_of_interest
        self._events_of_interest=events_of_interest
        self._continuous_of_interest=continuous_of_interest
        self._neurons_of_interest=neurons_of_interest
        self._markers_of_interest=markers_of_interest
        self._centroids_of_interest=centroids_of_interest
        self._clear_Nex_data=clear_Nex_data
        self.data=reader.ReadNex5File(file)
        self.FP, self.FP_sample_rate=self._pull_fp(FP_of_interest)
        self.events=self._pull_events(events_of_interest)
        self.continuous=self._pull_continuous(continuous_of_interest)
        self.neurons=self._pull_neurons(neurons_of_interest)
        self.markers=self._pull_markers(markers_of_interest)
        self.centroids=self._pull_centroids(centroids_of_interest)
        if clear_Nex_data:
            self.clear_fileData()
    
    def __str__(self):
        return 'File path:\n{} \n\nField potentials:\n{} \n\nEvents:\n{} \n\nContinuous:\n{} \n\nNeurons:\n{} \n\nMarkers:\n{} \n\nCentroids:\n{}'.format(self.file_path, \
        list(self.FP.columns), list(self.events.keys()),  list(map(lambda x: x.columns[0], self.continuous)), list(self.neurons.keys()), list(self.markers.keys()), list(self.centroids.columns))
    
    def __repr__(self):
        return 'NexData("{}",{}, {}, {}, {}, {}, {}, {})'.format(self.file_path, self._FP_of_interest, self._events_of_interest, self._continuous_of_interest, self._neurons_of_interest, \
                                                                 self._markers_of_interest, self._centroids_of_interest, self._clear_Nex_data)
    
    def _pull_fp(self, FP_of_interest=[]):
        return pull_fp(self.data, FP_of_interest)
    
    def _pull_events(self, events_of_interest=[]):
        return pull_events(self.data, events_of_interest)
    
    def _pull_continuous(self, continuous_of_interest=[]):
        return pull_continuous(self.data, continuous_of_interest)
    
    def _pull_neurons(self, neurons_of_interest=[]):
        return pull_neurons(self.data, neurons_of_interest)
    
    def _pull_markers(self, markers_of_interest=[]):
        return pull_markers(self.data, markers_of_interest)
    
    def _pull_centroids(self, centroids_of_interest=[]):
        return pull_centroids(self.data, centroids_of_interest)
    
    def clear_fileData(self):
        self.data = None