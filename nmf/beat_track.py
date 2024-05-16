from typing import Callable
import numpy as np
import lxml.etree

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


def nextpow2(n: int) -> int:
    n = int(n)
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def variable_stft(y: np.ndarray, bounds: np.ndarray, win_func: Callable = None):
    assert y.ndim == 1
    assert bounds.shape[1] == 2
    assert bounds.dtype == int

    nfft = nextpow2(np.max(np.diff(bounds, axis=1)))
    spec = np.zeros((1 + nfft // 2, bounds.shape[0]), dtype=complex)
    for i, (left, right) in enumerate(bounds):
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
