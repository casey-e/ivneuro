# -*- coding: utf-8 -*-
__doc__="""

ivneuro
=======
ivneuro is a Python package for processing and analyzing in-vivo neural and behavioral data.
It is optimized to analyze multiple variables of the same type, e.g. peri-event histograms of
simultaneusly recorded multiple local field potentials locked to multiple events.

"""

from ivneuro import events
from ivneuro import tracking
from ivneuro import continuous
from ivneuro import spectral
from ivneuro import pull_data
from ivneuro.events.utils import generate_signal