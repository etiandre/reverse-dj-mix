from typing import Callable
import numpy as np
import enum
from activation_learner import ActivationLearner
import scipy.stats
import matplotlib.pyplot as plt


def center_of_mass_columns(matrix: np.ndarray):
    indices = np.arange(matrix.shape[0])
    return np.sum(indices[:, np.newaxis] * matrix, axis=0) / np.sum(matrix, axis=0)


def rel_error(est: np.ndarray, real: np.ndarray):
    est = np.array(est)
    real = np.array(real)
    mask = ~np.isnan(est) & ~np.isnan(real)
    mre = np.mean(np.abs(est[mask] - real[mask]) / np.abs(real[mask]))
    return mre


def error(est: np.ndarray, real: np.ndarray):
    est = np.array(est)
    real = np.array(real)
    mask = ~np.isnan(est) & ~np.isnan(real)
    abs_error = np.mean(np.abs(est[mask] - real[mask]))
    return abs_error


def _apply_Hi(model: ActivationLearner, fn: Callable):
    ret = []
    for left, right in zip(model.split_idx, model.split_idx[1:]):
        H_track = model.H[left:right, :]
        ret.append(fn(H_track))
    return np.array(ret).T


class GainEstimator(enum.Enum):
    @enum.member
    @staticmethod
    def SUM(model):
        return _apply_Hi(model, fn=lambda Hi: np.sqrt(np.sum(Hi, axis=0)))

    @enum.member
    @staticmethod
    def MAX(model):
        return _apply_Hi(model, fn=lambda Hi: np.sqrt(np.max(Hi, axis=0)))

    @enum.member
    @staticmethod
    def RELSUM(model):
        ret = []
        for left, right in zip(model.split_idx, model.split_idx[1:]):
            H_track = model.H[left:right, :]
            ret.append(np.sqrt(np.sum(H_track, axis=0) / np.sum(model.H, axis=0)))
        return np.array(ret).T


class WarpEstimator(enum.Enum):
    @enum.member
    @staticmethod
    def CENTER_OF_MASS(model):
        return _apply_Hi(model, fn=center_of_mass_columns)

    @enum.member
    @staticmethod
    def ARGMAX(model):
        return _apply_Hi(model, fn=lambda Hi: np.argmax(Hi, axis=0))


def dynamic_threshold_moving_average(signal, window_size):
    threshold = np.convolve(
        np.abs(signal), np.ones(window_size) / window_size, mode="same"
    )
    return np.where(signal > threshold, 1, 0)


def estimate_highparams(tau, gain, warp, filter_size=0.1, plot=False):
    # normalize
    gain_norm = gain / gain.max()

    # median filter
    kernel_size = int(filter_size / (tau[1] - tau[0]))
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd
    gain_filt = scipy.signal.medfilt(gain_norm, kernel_size=kernel_size)

    # threshold and find contiguous playing slices
    thresh_gain_mask = np.ma.masked_array(gain_filt)
    thresh_gain_mask[gain_filt > 0.5] = np.ma.masked
    playing_slices = np.ma.clump_masked(thresh_gain_mask)

    if len(playing_slices) > 0:
        # get the longest one
        longest_slice = max(playing_slices, key=lambda i: i.stop - i.start)
    else:
        # thresholding failed: take the whole signal and hope for the best
        longest_slice = slice(0, len(gain) - 1)

    # pad
    longest_slice_len = longest_slice.stop - longest_slice.start
    rough_start_idx = max(longest_slice.start - longest_slice_len // 2, 0)
    rough_stop_idx = min(longest_slice.stop + longest_slice_len // 2, len(gain) - 1)

    # define ideal gain curve function
    def ideal_gain(tau, tau0, tau1, tau2, tau3):
        return np.piecewise(
            tau,
            [
                tau < tau0,
                (tau >= tau0) & (tau < tau1),
                (tau >= tau1) & (tau < tau2),
                (tau >= tau2) & (tau < tau3),
                tau >= tau3,
            ],
            [
                lambda tau: 0,
                lambda tau: (1 - 0) / (tau1 - tau0) * (tau - tau0) + 0,
                lambda tau: 1,
                lambda tau: (0 - 1) / (tau3 - tau2) * (tau - tau2) + 1,
                lambda tau: 0,
            ],
        )

    # fit ideal gain to signal
    tstart, tstop = tau[rough_start_idx], tau[rough_stop_idx]
    p, e = scipy.optimize.curve_fit(
        ideal_gain,
        tau[longest_slice],
        gain[longest_slice],
        p0=[
            tstart,
            tstart + (tstop - tstart) * 1 / 3,
            tstart + (tstop - tstart) * 2 / 3,
            tstop,
        ],
        bounds=(tstart, tstop),
    )

    # calculate fade bounds and slopes
    fadein_start, fadein_stop, fadeout_start, fadeout_stop = p
    fadein_slope = -1 / (fadein_start - fadein_stop)
    fadeout_slope = 1 / (fadeout_start - fadeout_stop)

    # use only the part of warp where gain is > 0
    mask = (tau > fadein_start) & (tau < fadeout_stop) & ~np.isnan(warp)
    warp_slope, warp_intercept, _, _, _ = scipy.stats.linregress(
        tau[mask], warp[mask]
    )

    # calculate track start time
    track_start = -warp_intercept / warp_slope

    if plot:
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(tau, gain_norm, label="norm", alpha=0.5)
        axes[0].plot(tau, gain_filt, label="filt")
        axes[0].plot(tau, ideal_gain(tau, *p), label="fit")
        axes[0].set_title("gain")
        axes[0].legend()

        axes[1].plot(tau, warp, label="input", alpha=0.5)
        axes[1].plot(tau[mask], warp[mask], label="masked")
        axes[1].plot(tau, warp_slope * tau + warp_intercept, label="fit")
        axes[1].set_title("warp")
        axes[1].legend()
    else:
        fig = None

    return (
        track_start,
        fadein_start,
        fadein_stop,
        fadein_slope,
        fadeout_start,
        fadeout_stop,
        fadeout_slope,
        fig,
    )
