import warnings
import os
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat
import hdf5storage

# Transformations
from timescales.sim import bin_spikes
from neurodsp.spectral import compute_spectrum
from timescales.autoreg import compute_ar_spectrum
from statsmodels.tsa.stattools import acf

# Workflow
from ndspflow.workflows import WorkFlow

# Models
from timescales.fit import PSD, ACF


# Paths
dirpath = '/Users/blancamartin/Desktop/Voytek_Lab/timescales/organoid-tango/hc-8/rat_organoid_timescale_development_paper/hc8_data/'
files = [f for f in sorted(os.listdir(dirpath)) if f.endswith('.npy')]

# Read .npy files
def reader(loc, dirpath=dirpath, files=files):
    """WorkFlow's npy reader function."""
    
    if isinstance(loc, (list, tuple, np.ndarray)):
        # Get a specific neurons of a single file
        file_ind, neuron_ind = loc
        file = f'{dirpath}{files[file_ind]}'
        spikes = np.load(file, allow_pickle=True)
        spikes = spikes[neuron_ind][0][0] 
    else:
        # Get all neurons of a single file
        file = f'{dirpath}{files[loc]}'
        spikes = np.load(file, allow_pickle=True)
        spikes = np.array([i[0][0] for i in spikes], dtype='object')
    
    return spikes

# Transformations
def compute_acf(spikes, nlags=200):
    
    corrs = acf(spikes, nlags=nlags, qstat=False, fft=True)[1:]
    lags = np.arange(1, len(corrs)+1)
    
    return lags, corrs


def bin_spikes(spike_times, bin_size, bins_per_window=None, fs=20000):
    
    # Convert from ms to samples
    bin_size = round((bin_size / 1000) * fs)
    samples = (spike_times / 1000 * fs).astype(int)

    # Init spike counts
    spikes = np.zeros(samples[-1] // bin_size, dtype=int)

    # Bin spikes
    bin_ind = 0

    for i in samples:

        bin_ind = i // bin_size

        if bin_ind > len(spikes)-1:
            break

        spikes[bin_ind] += 1
        
    # Create windows from binned counts
    if bins_per_window is not None:
        max_bin = int((len(spikes) // bins_per_window) * bins_per_window)
        spikes = spikes[:max_bin]
        spikes = spikes.reshape(-1, bins_per_window)
        
    return spikes


def compute_ar_spectrum_(spikes, fs, ar_order, f_range=None):
    """Prevent annoying warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, powers = compute_ar_spectrum(spikes, fs, ar_order, f_range=f_range)
    
    return freqs, powers

def mean(x, y):
    """Mean psd per window."""
    return x, np.nanmean(y, axis=0)



# def mean(x, y):
#     """Mean psd per window."""
    
#     y_out = np.nanmean(y, axis=0)
#     if np.any(np.isnan(y_out)):
#         print(y_out)
#         breakme
    
#     return x, y_out