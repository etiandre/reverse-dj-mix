from typing import Callable, Optional
import warnings
import librosa
import numpy as np
import lxml.etree
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm


# def variable_win_stft(
#     y: np.ndarray,
#     boundaries: np.ndarray,
#     nfft: Optional[int] = None,
#     win: Optional[np.ndarray] = None,
# ):
#     if nfft is None:
#         max_slice_len = np.max(np.diff(boundaries))
#         nfft = max_slice_len  # TODO: nextpow2
#     if max_slice_len > nfft:
#         warnings.warn(f"{max_slice_len=} is less than {nfft=}: will truncate")

#     spec = np.zeros((1 + nfft // 2, boundaries.shape[0] - 1), dtype=complex)
#     for i, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
#         col = np.fft.rfft(y[left:right], nfft)
#         spec[:, i] = col

#     if win is not None:
#         spec = scipy.ndimage.convolve1d(spec, win, axis=1, mode="constant")
#     return spec, nfft

def variable_stft(y: np.ndarray, boundaries: np.ndarray, win_func: Callable=None):
    assert y.ndim == 1
    assert boundaries.shape[1] == 2
    assert boundaries.dtype == int
    
    nfft = np.max(np.diff(boundaries, axis=1))
    spec = np.zeros((1 + nfft // 2, boundaries.shape[0]), dtype=complex)
    for i, (left, right) in enumerate(tqdm(boundaries, desc="STFT")):
        if win_func is not None:
            col = np.fft.rfft(y[left:right] * win_func(right - left), nfft)
        else:
            col = np.fft.rfft(y[left:right], nfft)
            
        spec[:, i] = col
    return spec, nfft

def parse_ircambeat_xml(xml: bytes):
    tree = lxml.etree.fromstring(xml)
    ns = {"music": "http://www.quaero.org/Music_6/1.0"}
    ret = []
    for beat in tree.xpath(
        "/music:musicdescription/music:segment/music:beattype", namespaces=ns
    ):
        time = float(beat.getparent().attrib["time"])
        ret.append(time)
    return np.array(ret)
