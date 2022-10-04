from timescales.autoreg import compute_ar_spectrum
from neurodsp.spectral import compute_spectrum
import numpy as np
import warnings
warnings.filterwarnings('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_dpdf(time_object, day_diff):
    """ Converts data time object into days post differentiation int value"""
    duration = time_object - day_diff
    duration_in_s = duration.total_seconds()
    # days post differentiation in days
    days = duration.days
    dpdf = int(divmod(duration_in_s, 86400)[0])      # Seconds in a day = 86400
    # add 56 days (8 weeks) to calculated day - first recording was done at 8 weeks
    # post differentiation
    dpdf = dpdf + 56

    return dpdf


def bin_spikes(spike_times, bin_size, bins_per_window=None, fs=20000, n_recording_bins=None):
    # Convert from ms to samples
    bin_size = round((bin_size / 1000) * fs)
    samples = (spike_times / 1000 * fs).astype(int)

    # Init spike counts
    if n_recording_bins is None:
        spikes = np.zeros(samples[-1] // bin_size, dtype=int)
    else:
        spikes = np.zeros(n_recording_bins, dtype=int)
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


# def trial_average_spectrum_welch(trial_data, fs, f_range=None):
#     power_all = []
#     for trial in trial_data:
#         print(np.array(trial))
#         freq, power = compute_spectrum(
#             np.array(trial), fs, method='welch', f_range=f_range)
#         power_all.append(power)
#     power_all_np = np.array(power_all)
#     power_avg = np.mean(power_all_np, axis=0)
#     return freq, power_avg


def trial_average_spectrum_welch(trial_data, fs, f_range=None):
    trial_data_np = np.array(trial_data)
    freq, power = compute_spectrum(
        trial_data_np, fs, method='welch', avg_type='median', f_range=f_range)
    power_avg = np.mean(power, axis=0)
    return freq, power_avg

# def trial_average_spectrum_welch(trial_data, fs, f_range=None):
#     trial_data_np = np.array(trial_data)
#     if trial_data_np.size > 0:
#         freq, power = compute_spectrum(
#             trial_data_np, fs, method='welch', avg_type='median', f_range=f_range)
#         power_avg = np.mean(power, axis=0)
#     else:
#         freq,power_avg = np.array(np.nan),np.array(np.nan)
#     return freq, power_avg


def compute_ar_spectrum_(spikes, fs, ar_order, f_range=None):
    """Prevent annoying warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, powers = compute_ar_spectrum(
            spikes, fs, ar_order, f_range=f_range)
    return freqs, powers


def trial_average_spectrum_ar(trial_data, fs, ar_order, f_range=None):
    power_all = []
    for trial in trial_data:
        freq, power = compute_ar_spectrum_(
            trial, fs, ar_order, f_range=f_range)
        power_all.append(power)
    power_all_np = np.array(power_all)
    power_avg = np.mean(power_all_np, axis=0)
    return freq, power_avg
