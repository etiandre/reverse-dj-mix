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
    return ax.imshow(X_, **kwargs)


def plot_carve_resize(H_carved: np.ndarray, H_carved_resized: np.ndarray):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("H carved")
    imshow_highlight_zero(H_carved, cmap="turbo", origin="lower", aspect="auto")
    plt.subplot(1, 2, 2)
    plt.title("H resized")
    imshow_highlight_zero(H_carved_resized, cmap="turbo", origin="lower", aspect="auto")
    return fig


def plot_loss_history(losses: list[dict], ax=None):
    ax = ax or plt.gca()
    for k in losses[0]["penalties"].keys():
        ax.plot([i["penalties"][k] for i in losses], label=k)
    ax.plot([i["divergence"] for i in losses], label="divergence")
    ax.plot([i["full"] for i in losses], "--", label="total")

    ax.set_xlabel("iter")
    ax.set_ylabel("losses")
    ax.set_yscale("log")
    ax.legend()

    return ax


def plot_warp(
    tau: np.ndarray,
    warps: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    ax=None,
):
    ax = ax or plt.gca()

    def plot_pos(positions, real: bool):
        for i, t in enumerate(positions.T):
            ax.plot(
                tau,
                t,
                "--" if real else "-",
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=f"track {i}" if not real else None,
            )

    plot_pos(warps, False)
    if ground_truth is not None:
        plot_pos(ground_truth, True)
    ax.legend()
    ax.set_xlabel("mix time (s)")
    ax.set_ylabel("ref frame")


def plot_gain(
    tau: np.ndarray,
    gains: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    ax=None,
):
    ax = ax or plt.gca()

    def plot_g(gains, real: bool):
        for i, g in enumerate(gains.T):
            ax.plot(
                tau,
                g,
                "--" if real else "-",
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=f"track {i}" if not real else None,
            )

    plot_g(gains, False)
    if ground_truth is not None:
        plot_g(ground_truth, True)
    ax.legend()
    ax.set_title("track gain")
    ax.set_xlabel("mix time (s)")
    # ax.set_ylim(0, 1)


# TODO: time in seconds
def plot_H(H: np.ndarray, split_idx=None, ax=None):
    ax = ax or plt.gca()
    im = imshow_highlight_zero(H, ax, cmap=CMAP, aspect="auto", origin="lower")

    if split_idx is not None:
        for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
            COLOR = "pink"
            ax.axhline(a - 0.5, color=COLOR, linestyle="--")
            ax.annotate(f"track {track}", (0, (a + b) / 2), color=COLOR)

    ax.set_xlabel("mix frame")
    ax.set_ylabel("ref frame")
    return im


# TODO: time in seconds
def plot_pow_spec(W: np.ndarray, split_idx=None, ax=None):
    ax = ax or plt.gca()
    im = _pow_specshow(W, ax)
    # annotate track boundaries if given
    if split_idx is not None:
        COLOR = "pink"
        for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
            ax.axvline(a - 0.5, color=COLOR, linestyle="--")
            ax.annotate(f"track {track}", ((a + b) / 2, 1), color=COLOR)

    return im


def plot_nmf(model: ActivationLearner):
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))

    im = plot_H(model.H, model.split_idx, ax=axes[0, 0])
    fig.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title("H")

    im = plot_pow_spec(model.W, model.split_idx, ax=axes[0, 1])
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title("W")

    im = plot_pow_spec(model.V, ax=axes[1, 0])
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("V")

    im = plot_pow_spec(model.W @ model.H, ax=axes[1, 1])
    fig.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title("WH")

    plt.tight_layout()
    return fig
