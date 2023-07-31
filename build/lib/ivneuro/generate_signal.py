# -*- coding: utf-8 -*-
"""
Function to generate a signal, used for examples of other functions of the package.
"""
import numpy as np
from .utils import generate_pink_noise, generate_oscillatory_signal, add_signal_to_noise

def generate_signal (duration, burst_timestamps, burst_frequency, burst_duration = 2, burst_amplitude = 0.1, sampling_frequency=1000, seed = 40):
    """
    Generate a signal with pink noise and increases in power (bursts) at a specified frequency.

    Parameters
    ----------
    duration : int or float
        Duration of the signal in seconds.
    burst_timestamps : list of floats
        Timestamps at wich the increases in power must occur.
    burst_frequency : int or float
        Frequency at wich the signal displays increases in power.
    burst_duration : int or float, optional
        Duration (in seconds) of high power burst. The default is 2.
    burst_amplitude : int or float, optional
        Amplitud of the signal used to create the increases in power. The default is 0.1.
    sampling_frequency : int, optional
        Sampling frequency. The default is 1000.
    seed = int, optional
    Value for np.random.seed to generate pink noise. The default is 40.

    Returns
    -------
    signal : pandas DataFrame
        Timestamps as index and signal values as values.

    """
    #Generate pink noise
    pink_noise=generate_pink_noise(duration, sampling_frequency, seed)
    pink_noise.index=np.round(pink_noise.index,3)
    
    # Generate burst signal
    burst_signal = generate_oscillatory_signal(frequency=burst_frequency, duration = burst_duration, amplitude = burst_amplitude, sample_rate=sampling_frequency)
    burst_signal.index=np.round(burst_signal.index, 3)
    
    #Add burst signal to pink_noise at the burst timestamps
    signal = add_signal_to_noise (pink_noise, burst_signal, burst_timestamps)
    
    return signal