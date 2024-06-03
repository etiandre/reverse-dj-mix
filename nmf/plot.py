from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import librosa
from activation_learner import ActivationLearner

COLOR_CYCLE = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
CMAP = "turbo"


def _pow_specshow(S, ax=None):
    ax = ax or plt.gca()
    return ax.imshow(
        librosa.power_to_db(S),
        cmap=CMAP,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )


def imshow_highlight_zero(X: np.ndarray, ax=None, **kwargs):
    ax = ax or plt.gca()
    X_ = X.copy()
    X_[X_ == 0] = np.nan
    ax.imshow(X_, **kwargs)


def plot_carve_resize(H_carved: np.ndarray, H_carved_resized: np.ndarray):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("H carved")
    imshow_highlight_zero(H_carved, cmap="turbo", origin="lower", aspect="auto")
    plt.subplot(1, 2, 2)
    plt.title("H resized")
    imshow_highlight_zero(H_carved_resized, cmap="turbo", origin="lower", aspect="auto")
    return fig


def plot_losses(losses: np.ndarray, ax=None):
    ax = ax or plt.gca()
    ax.set_xlabel("iter")
    ax.set_ylabel("losses")
    for i, loss_component in enumerate(losses):
        ax.plot(loss_component, label=f"loss {i}")
    ax.set_yscale("log")
    ax.legend()


def plot_timeremap(
    positions: np.ndarray,
    hop_size: float,
    ground_truth: Optional[np.ndarray] = None,
    ax=None,
):
    ax = ax or plt.gca()

    def plot_pos(positions, real: bool):
        for i, ref_time in enumerate(positions.T):
            mix_time = np.arange(len(ref_time)) * hop_size
            coords = np.vstack([mix_time, ref_time]).T
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    "--" if real else "-",
                    color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                )

    plot_pos(positions, False)
    if ground_truth is not None:
        plot_pos(ground_truth, True)
    ax.set_xlabel("mix time (s)")
    ax.set_ylabel("ref time (s)")


def plot_volume(
    volumes: np.ndarray,
    hop_size: float,
    ground_truth: Optional[np.ndarray] = None,
    ax=None,
):
    ax = ax or plt.gca()

    def plot_vol(volumes, real: bool):
        for i, v in enumerate(volumes.T):
            mix_time = np.arange(len(v)) * hop_size
            ax.plot(
                mix_time,
                v,
                "--" if real else "-",
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=f"track {i}" if real else None,
            )

    plot_vol(volumes, False)
    if ground_truth is not None:
        plot_vol(ground_truth, True)
    ax.legend()
    ax.set_title("track volume")
    ax.set_xlabel("mix time (s)")
    # ax.set_ylim(0, 1)


# TODO: time in seconds
def plot_H(H: np.ndarray, split_idx, ax=None):
    ax = ax or plt.gca()
    im = ax.imshow(H, cmap=CMAP, aspect="auto", origin="lower")

    for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
        ax.axhline(a - 0.5, color="r", linestyle="--")
        ax.annotate(f"track {track}", (0, (a + b) / 2), color="red")

    ax.set_title("H (activations)")
    ax.set_xlabel("mix frame")
    ax.set_ylabel("ref frame")
    return im


# TODO: time in seconds
def plot_pow_spec(W: np.ndarray, split_idx=None, ax=None):
    ax = ax or plt.gca()
    im = _pow_specshow(W, ax)
    if split_idx is not None:
        for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
            ax.axvline(a - 0.5, color="r", linestyle="--")
            ax.annotate(f"track {track}", ((a + b) / 2, 1), color="red")

    return im


def plot_nmf(model: ActivationLearner):
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))

    im = plot_H(model.H, model.split_idx, ax=axes[0, 0])
    fig.colorbar(im, ax=axes[0, 0])

    im = plot_pow_spec(model.W, model.split_idx, ax=axes[0, 1])
    fig.colorbar(im, ax=axes[0, 1])

    im = plot_pow_spec(model.V, ax=axes[1, 0])
    fig.colorbar(im, ax=axes[1, 0])

    im = plot_pow_spec(model.W @ model.H, ax=axes[1, 1])
    fig.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    return fig
