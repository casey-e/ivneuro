# -*- coding: utf-8 -*-
__doc__="""

ivneuro
=======
Tools for processing and analyzing neurophysiological signals recorded in-vivo.


ivneuro provides tools for analyzing neural signals recorded in-vivo during behavior. It 
focus on time series analyses of continuous variables such as Local Field Potentials and 
is optimized to process either single signals in a single condition as well as multiple 
signals in multiple conditions simultaneously. It also provides a subpackage for extracting 
data from Nex files.

"""

from .events import make_intervals, classify_events_base_on_time
from .tracking import scale_centroids, calculate_speed, position_of_event, distances_to_position, EventPosition
from .continuous import calculate_sampling_period, calculate_sampling_rate, peh, PeriEventHistogram
from .spectograms import peri_event_spectogram, normalize_pes, plot_pes, PeriEventSpectogram
from .delta_power_spectral import delta_power_spectral
from .delta_coherence import delta_coherence
from . import nex
from .generate_signal import generate_signal