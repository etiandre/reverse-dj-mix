from typing import Callable
import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import enum
from activation_learner import ActivationLearner


def center_of_mass_columns(matrix):
    indices = np.arange(matrix.shape[0])
    return np.sum(indices[:, np.newaxis] * matrix, axis=0) / np.sum(matrix, axis=0)


def rel_error(est: np.ndarray, real: np.ndarray):
    mask = ~np.isnan(est) & ~np.isnan(real)
    mre = np.mean(np.abs(est[mask] - real[mask]) / np.abs(real[mask]))
    return mre


def _estimate(model: ActivationLearner, fn: Callable):
    ret = []
    for left, right in zip(model.split_idx, model.split_idx[1:]):
        H_track = model.H[left:right, :]
        ret.append(fn(H_track))
    return np.array(ret).T


class VolumeEstimator(enum.Enum):
    @enum.member
    @staticmethod
    def SUM(model):
        return _estimate(model, fn=lambda Hi: np.sqrt(np.sum(Hi, axis=0)))

    @enum.member
    @staticmethod
    def MAX(model):
        return _estimate(model, fn=lambda Hi: np.sqrt(np.max(Hi, axis=0)))


class TimeRemappingEstimator(enum.Enum):
    @enum.member
    @staticmethod
    def CENTER_OF_MASS(model):
        return _estimate(model, fn=center_of_mass_columns)

    @enum.member
    @staticmethod
    def ARGMAX(model):
        return _estimate(model, fn=lambda Hi: np.argmax(Hi, axis=0))
