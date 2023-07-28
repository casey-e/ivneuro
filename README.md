# ivneuro


**Tools for processing and analyzing neurophysiological signals recorded *in-vivo*.**


**ivneuro** provides tools for analyzing neural signals dinamycs recorded *in-vivo* during behavior. It focus on the performance of high-level timeseries analysis of continuous variables such as Local Field Potentials. It is optimized to process either single signals in a single condition as well as multiple signals in multiple conditions simultaneusly (as in the example below). 

Installation
-----------
To install use:
```
pip install ivneuro
```


Documentation
-------------
Documentation can be found [here]()

Contributing
------------
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are strongly appreciated. To contribute, please contact the author to e.toccalino@gmail.com

Quick examples
-------------
**Peri-event histograms**
```
>>> # Create events
>>> burst = [*range(30,300, 30)]
>>> control = [*range(45,300, 30)]
>>> events={'burst':burst,'control':control}

>>> # Generate signals
>>> signal1 = ivneuro.generate_signal(300, burst, 2, burst_duration = 0.5, burst_amplitude=1)
>>> signal2 = ivneuro.generate_signal(300, burst, 0.5, burst_duration = 2, burst_amplitude=1.5, seed = 21)
>>> signal = pd.concat([signal1, signal2], axis = 1)


>>> # Make peri-event histogram
>>> hist=ivneuro.continuous.peh(signal, events, lower_lim = -5, higher_lim = 5)

>>> hist.plot()
```
![Alt text](image.png)

<br>

**Peri-event spectograms**
```
>>> # Create events
>>> burst = [*range(30,300, 30)]
>>> control = [*range(45,300, 30)]
>>> events={'burst':burst,'control':control}

>>> # Generate signals
>>> signal1 = ivneuro.generate_signal(300, burst, 30, burst_amplitude=0.06)
>>> signal2 = ivneuro.generate_signal(300, burst, 32, burst_amplitude=0.13)
>>> signal3 = ivneuro.generate_signal(300, burst, 80, burst_amplitude=0.05)
>>> signals = pd.concat([signal1, signal2, signal3], axis = 1)

>>> # Calculate peri-event spectogram
>>> pes = ivneuro.spectral.peri_event_spectogram(signals, events, -10, 10, higher_freq=100)
>>> pes.normalize().plot()
```
![Alt text](image-1.png)

<br>

**Difference in power spectral between 2 conditions**
```
>>> # Create event
>>> burst = [*range(30,300, 30)]

>>> # Make intervals
>>> burst_intervals = ivneuro.events.make_intervals(burst, 0, 2)
>>> baseline_intervals = ivneuro.events.make_intervals(burst, -6, -4)

>>> # Generate signals
>>> signal1 = ivneuro.generate_signal(300, burst, 30, burst_amplitude=0.06, seed=15)
>>> signal2 = ivneuro.generate_signal(300, burst, 80, burst_amplitude=0.05, seed = 50)
>>> signals = pd.concat([signal1, signal2], axis = 1)


>>> # Calculate change in power spectral
>>> power_burst, power_baseline, delta_power = ivneuro.spectral.delta_power_spectral(signals, burst_intervals, baseline_intervals, lowest_freq = 0, highest_freq = 100)
>>> delta_power.plot(xlabel = 'Frequency (Hz)', ylabel = 'Normalized change in power')
```
![Alt text](image-2.png)


Licence
-------
The codes are released under [MIT License](https://mit-license.org/)
