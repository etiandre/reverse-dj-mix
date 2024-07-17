import numpy as np
import scipy.signal
import pydub

def fade(a: np.ndarray, b: np.ndarray, l: int):
    """
    linear fade between 2 signals
    a: signal 1
    b: signal 2
    l: fade duration in samples
    """
    curve = np.linspace(0, 1, l)
    a_fade = a[-l:] * (1 - curve)
    b_fade = b[:l] * curve
    return np.concatenate([a[:-l], a_fade + b_fade, b[l:]])


def lowpass_filter(data, cutoff_freq, sampling_rate, order=1):
    """
    Apply a 1st order lowpass filter to the data.

    Parameters:
    - data: numpy array containing the signal to be filtered
    - cutoff_freq: cutoff frequency in Hz
    - sampling_rate: sampling rate in Hz
    - order: order of the filter

    Returns:
    - filtered_data: numpy array containing the filtered signal
    """
    # Normalize the cutoff frequency
    normal_cutoff = cutoff_freq / (0.5 * sampling_rate)

    # Design the Butterworth filter
    sos = scipy.signal.butter(
        order, normal_cutoff, btype="low", analog=False, output="sos"
    )

    # Apply the filter to the data
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def write_mp3(path, sr, x, normalized=True):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(path, format="mp3", bitrate="320k")
