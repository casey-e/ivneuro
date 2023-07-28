# -*- coding: utf-8 -*-
__doc__="""
ivneuro.tracking
================
A subpackage for processing and analyze positions data based on xy coordinates.
"""

from ivneuro.tracking.tracking import scale_centroids, calculate_speed, position_of_event, distances_to_position
from ivneuro.tracking.EventPosition import EventPosition