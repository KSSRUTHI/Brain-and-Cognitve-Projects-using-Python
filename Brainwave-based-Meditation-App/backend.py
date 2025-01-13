import numpy as np
import pandas as pd
from mne import filter
import sounddevice as sd
import os

BAND_RANGES = {
    "alpha": (8, 12),
    "beta": (13, 30),
    "theta": (4, 7),
}
SFREQ = 256  

def generate_synthetic_eeg(channels=4, samples=256):
    return np.random.randn(channels, samples)

def filter_band(eeg_data, sfreq, band):
    l_freq, h_freq = BAND_RANGES[band]
    return filter.filter_data(eeg_data, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)


def compute_band_power(band_wave_data):
    return np.mean(np.square(band_wave_data))


def save_session_data(data, filename="synthetic_eeg_session.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return os.path.abspath(filename)
