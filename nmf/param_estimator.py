from typing import Callable
import numpy as np
import enum
from activation_learner import ActivationLearner
import scipy.stats
import matplotlib.pyplot as plt


def center_of_mass_columns(matrix: np.ndarray):
    indices = np.arange(matrix.shape[0])
    colsum = np.sum(matrix, axis=0)
    colsum[colsum == 0] = 1  # TODO: hack
    return np.sum(indices[:, np.newaxis] * matrix, axis=0) / colsum


def rel_error(est: np.ndarray, real: np.ndarray):
    est = np.array(est)
    real = np.array(real)
    mask = ~np.isnan(est) & ~np.isnan(real)
    mre = np.mean(np.abs(est[mask] - real[mask]) / np.abs(real[mask]))
    return float(mre)


def error(est: np.ndarray, real: np.ndarray):
    est = np.array(est)
    real = np.array(real)
    mask = ~np.isnan(est) & ~np.isnan(real)
    abs_error = np.mean(np.abs(est[mask] - real[mask]))
    return float(abs_error)


def _apply_Hi(model: ActivationLearner, fn: Callable):
    ret = []
    for left, right in zip(model.split_idx, model.split_idx[1:]):
        H_track = model.H[left:right, :]
        ret.append(fn(H_track))
    return np.array(ret).T


class GainEstimator(enum.Enum):
    # @enum.member
    @staticmethod
    def SUM(model):
        return _apply_Hi(model, fn=lambda Hi: np.sqrt(np.sum(Hi, axis=0)))

    # @enum.member
    # @staticmethod
    # def MAX(model):
    #     return _apply_Hi(model, fn=lambda Hi: np.sqrt(np.max(Hi, axis=0)))


class WarpEstimator(enum.Enum):
    # @enum.member
    @staticmethod
    def CENTER_OF_MASS(model, hop_size: float):
        return _apply_Hi(model, fn=center_of_mass_columns) * hop_size

    # @enum.member
    # @staticmethod
    # def ARGMAX(model):
    #     return _apply_Hi(model, fn=lambda Hi: np.argmax(Hi, axis=0))


def ideal_gain(tau, tau0, a, b, c, g_max):
    tau1 = tau0 + a
    tau2 = tau1 + b
    tau3 = tau2 + c
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
            lambda tau: (g_max - 0) / (tau1 - tau0) * (tau - tau0) + 0,
            lambda tau: g_max,
            lambda tau: (0 - g_max) / (tau3 - tau2) * (tau - tau2) + g_max,
            lambda tau: 0,
        ],
    )


def fit_ideal_gain(tau, gain):
    start_bounds = [tau[0], tau[-1]]
    fadein_duration_bounds = [1, 10]
    play_duration_bounds = [1, np.inf]
    fadeout_duration_bounds = [1, 10]
    gmax_bounds = [0.1, 1]
    # fit ideal gain to signal
    p, e = scipy.optimize.curve_fit(
        ideal_gain,
        tau,
        gain,
        bounds=np.column_stack(
            [
                start_bounds,
                fadein_duration_bounds,
                play_duration_bounds,
                fadeout_duration_bounds,
                gmax_bounds,
            ]
        ),
    )

    # calculate fade bounds and slopes
    tau0, a, b, c, g_max = p
    fadein_start = tau0
    fadein_stop = tau0 + a
    fadeout_start = tau0 + a + b
    fadeout_stop = tau0 + a + b + c
    return (
        fadein_start,
        fadein_stop,
        fadeout_start,
        fadeout_stop,
        0,
        g_max,
    )


def estimate_highparams(tau, gain, warp, filter_size=5, plot=False):
    # median filter
    kernel_size = int(filter_size / (tau[1] - tau[0]))
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd
    gain_filt = scipy.signal.medfilt(gain, kernel_size=kernel_size)

    # normalize
    gain_norm = gain_filt / gain_filt.max()

    # threshold and find contiguous playing slices
    thresh_gain_mask = np.ma.masked_array(gain_norm)
    thresh_gain_mask[gain_norm < 0.5] = np.ma.masked
    playing_slices = np.ma.clump_unmasked(thresh_gain_mask)

    if len(playing_slices) > 0:
        # get the longest one
        longest_slice = max(playing_slices, key=lambda i: i.stop - i.start)
    else:
        # thresholding failed: take the whole signal and hope for the best
        longest_slice = slice(0, len(gain) - 1)

    # pad the slice
    longest_slice_len = longest_slice.stop - longest_slice.start
    rough_start_idx = max(longest_slice.start - longest_slice_len // 2, 0)
    rough_stop_idx = min(longest_slice.stop + longest_slice_len // 2, len(gain) - 1)
    longest_slice = slice(rough_start_idx, rough_stop_idx)
    (
        fadein_start,
        fadein_stop,
        fadeout_start,
        fadeout_stop,
        g_min,
        g_max,
    ) = fit_ideal_gain(tau[longest_slice], gain_norm[longest_slice])
    # use only the part of warp where gain is > 0
    mask = (tau > fadein_start) & (tau < fadeout_stop) & ~np.isnan(warp)
    speed, warp_intercept, _, _, _ = scipy.stats.linregress(tau[mask], warp[mask])

    # calculate track start time
    track_start = -warp_intercept / speed

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(20,6))
        axes[0].plot(tau, gain, label="input", alpha=0.5)
        axes[0].plot(tau, gain_norm, label="filt")
        axes[0].plot(tau, thresh_gain_mask, label="thresh")
        axes[0].axvline(tau[longest_slice.start], linestyle='--')
        axes[0].axvline(tau[longest_slice.stop], linestyle='--')
        axes[0].plot(
            [fadein_start, fadein_stop, fadeout_start, fadeout_stop],
            [g_min, g_max, g_max, g_min],
            label="fit",
        )
        axes[0].set_title("gain")
        axes[0].legend()

        axes[1].plot(tau, warp, label="input", alpha=0.5)
        axes[1].plot(tau[mask], warp[mask], label="masked")
        axes[1].plot(tau[mask], speed * tau[mask] + warp_intercept, label="fit")
        axes[1].set_title("warp")
        axes[1].legend()
    else:
        fig = None

    return (
        track_start,
        fadein_start,
        fadein_stop,
        fadeout_start,
        fadeout_stop,
        speed,
        fig,
    )
