import numpy as np
from tqdm import tqdm


def resample(x: np.ndarray, tau: np.ndarray, T: float) -> np.ndarray:
    """Resamples with variable rate

    Args:
        x (np.ndarray): signal to resample
        tau (np.ndarray): vector of desired times
        T (float): sampling period of x

    Returns:
        np.ndarray: resampled signal
    """
    y = np.zeros_like(tau)
    n = np.arange(len(x))
    for m in tqdm(range(len(y))):
        y[m] = np.sum(x * np.sinc((tau[m] - n * T) / T))

    return y


def resample_fast(
    x: np.ndarray, new_indices: np.ndarray, halfwlen: int = 100
) -> np.ndarray:
    """Resamples with variable rate (truncated window)

    Args:
        x (np.ndarray): signal to resample
        tau (np.ndarray): vector of desired times
        T (float): sampling period of x
        halfwlen (int): half length of window

    Returns:
        np.ndarray: resampled signal
    """
    y = np.zeros_like(new_indices)
    for m in tqdm(range(len(y))):
        kmin = max(int(round(new_indices[m])) - halfwlen, 0)
        kmax = min(int(round(new_indices[m])) + halfwlen, len(x))
        k = np.arange(kmin, kmax)
        w = np.sinc(new_indices[m] - k)
        y[m] = np.sum(x[k] * w)

    return y
