def fade(a: np.ndarray, b: np.ndarray, l: int):
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