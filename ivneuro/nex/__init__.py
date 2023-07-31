# -*- coding: utf-8 -*-
__doc__="""
ivneuro.nex
=================
A subpackage for extracting data from nex5 files.
It can also extract centroids and local field potentials, which do not have their own types in
NeuroExplorer, as a different type from continuous.
The NexData class puts together all the functions of this subpackage and creates a NexData object,
wich contains the extrated variables as attributes.
"""

from .pull_nex_data import pull_fp, pull_events, pull_continuous, pull_neurons, pull_markers, pull_centroids
from .NexData import NexData