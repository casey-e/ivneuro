# -*- coding: utf-8 -*-
__doc__="""
ivneuro.spectral
================
A subpackage for spectral analyses.

"""

from ivneuro.spectral.spectograms import peri_event_spectogram, normalize_pes, plot_pes, PeriEventSpectogram
from ivneuro.spectral.delta_power_spectral import delta_power_spectral
from ivneuro.spectral.delta_coherence import delta_coherence