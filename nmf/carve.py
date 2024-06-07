import numpy as np
import scipy.sparse
import cv2


def carve(H: np.ndarray, split_idx, threshold_dB):
    threshold = 10 ** (threshold_dB / 20)
    for left, right in zip(split_idx, split_idx[1:]):
        energy = H[left:right, :].max(axis=0)
        H[left:right, energy < threshold] = 0
    return H


def carve_naive(H: np.ndarray, threshold_dB):
    threshold = 10 ** (threshold_dB / 20)
    H[H < threshold] = 0
    return H


def resize_cv_area(img: np.ndarray, shape: tuple[int, int]):
    return cv2.resize(
        img,
        dsize=(shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
