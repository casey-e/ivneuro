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

