# -*- coding: utf-8 -*-
"""
A module with private helper functions.
"""

import numpy as np
import pandas as pd



def significant_decimal_positions (value):
    """
    Calculate the significant number of decimal positions to use to round a value.
    
    Certains calculations, like vectorized operations that switch between decimal and binary systems, generate values with large amount of
    decimal positions that can interfiere with posterior operations (like aligments). This function keeps only the relevant amount of decimal 
    positions.

    Parameters
    ----------
    value : float
        Value to calculate the number of significant decimal positions.

    Returns
    -------
    int
        Number of significant decimal positions, to be used as imput of numpy.round() .
    """
    
    return int(np.ceil(abs(np.log10(value))))


def generate_pink_noise(duration, sampling_frequency, seed): #Function to generate pink noise
    np.random.seed(seed)
    num_samples = int(duration * sampling_frequency)
    pink_noise = np.empty(num_samples)
    
    # Number of accumulators, should be a power of 2
    num_accumulators = 16
    
    # The first accumulator
    accumulator = np.zeros(num_accumulators)
    
    for i in range(num_samples):
        # Generate a random value between -1 and 1
        white_noise = np.random.uniform(-1, 1)
        
        # Update the accumulators and add the current value to the pink noise
        accumulator = 0.997 * accumulator + white_noise
        pink_noise[i] = accumulator.sum()
    
    # Scale the pink noise to have a standard deviation of 1
    pink_noise /= np.std(pink_noise)
    
    # Create timestamps
    time = np.arange(0, duration, 1/sampling_frequency)
    
    return pd.DataFrame(index=time, data=pink_noise, columns=['Pink_noise'])

def generate_oscillatory_signal(frequency, duration, amplitude, sample_rate=1000): # Function to generate a signal
    time = np.linspace(0, duration, int(sample_rate * duration))
    signal = amplitude * np.sin(2 * np.pi * frequency * time)
    
    return pd.DataFrame(index = time, data = signal, columns = [str(frequency)+'Hz'])

def add_signal_to_noise (noise, signal, timestamps): # Function to add a signal to a noise at specific timestamps
    # Create signal repeated across timestamps
    repeated_signal=[signal.set_index(signal.index + i) for i in timestamps]
    repeated_signal =pd.concat(repeated_signal)
    
    # Add signal to noise
    result=pd.concat([noise,repeated_signal], axis=1)
    result=result[~result.iloc[:,0].isnull()]
    result=pd.DataFrame(result.iloc[:,0].add(result.iloc[:,1], fill_value=0), columns=['Signal '+signal.columns[0]])
    
    return result

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