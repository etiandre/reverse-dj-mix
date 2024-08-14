from re import split
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import librosa
import matplotlib.patheffects as PathEffects

COLOR_CYCLE = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
CMAP = "turbo"
IGNORED_COLOR = "pink"
IGNORED_ALPHA = 0.5
TRACK_BOUNDARY_COLOR = "white"
TRACK_BOUNDARY_PATHEFFECTS = [PathEffects.withStroke(linewidth=3, foreground="black")]


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
    im = imshow_highlight_zero(H_carved, cmap="turbo", origin="lower", aspect="auto")
    fig.colorbar(im, ax=plt.gca())  # Add colorbar to the first subplot
    plt.subplot(1, 2, 2)
    plt.title("H resized")
    im = imshow_highlight_zero(
        H_carved_resized, cmap="turbo", origin="lower", aspect="auto"
    )
    fig.colorbar(im, ax=plt.gca())  # Add colorbar to the second subplot
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
                "--" if real else "x",
                color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                label=f"track {i}" if not real else None,
                alpha=1 if real else 0.7,
            )

    plot_pos(warps, False)
    if ground_truth is not None:
        plot_pos(ground_truth, True)
    ax.legend()
    ax.set_xlabel("mix time (s)")
    ax.set_ylabel("track time (s)")


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
                "--" if real else "x",
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


def plot_H(H: np.ndarray, split_idx=None, ignored_lines=None, ax=None):
    ax = ax or plt.gca()
    im = imshow_highlight_zero(H, ax=ax, cmap=CMAP, aspect="auto", origin="lower")

    if split_idx is not None:
        for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
            ax.axhline(
                a - 0.5,
                color=TRACK_BOUNDARY_COLOR,
                linestyle="--",
                path_effects=TRACK_BOUNDARY_PATHEFFECTS,
            )
            if track == len(split_idx) - 2:
                ax.axhline(
                    b - 0.5,
                    color=TRACK_BOUNDARY_COLOR,
                    linestyle="--",
                    path_effects=TRACK_BOUNDARY_PATHEFFECTS,
                )
            ax.annotate(
                f"{track+1}",
                (0, (a + b) / 2),
                color=TRACK_BOUNDARY_COLOR,
                path_effects=TRACK_BOUNDARY_PATHEFFECTS,
                annotation_clip=False,
                rotation="vertical",
                horizontalalignment="center",
                rotation_mode="anchor",
            )

    if ignored_lines is not None:
        for l in ignored_lines.nonzero(as_tuple=True)[1]:
            ax.fill_between(
                (-0.5, H.shape[1] - 0.5),
                l - 0.5,
                l,
                color=IGNORED_COLOR,
                alpha=IGNORED_ALPHA,
            )

    ax.set_xlabel("mix frame")
    ax.set_ylabel("ref frame")
    return im


# TODO: time in seconds
def plot_pow_spec(W: np.ndarray, split_idx=None, ignored_cols=None, ax=None):
    ax = ax or plt.gca()
    im = _pow_specshow(W, ax)
    # annotate track boundaries if given
    if split_idx is not None:
        for track, (a, b) in enumerate(zip(split_idx, split_idx[1:])):
            ax.axvline(
                a - 0.5,
                color=TRACK_BOUNDARY_COLOR,
                linestyle="--",
                path_effects=TRACK_BOUNDARY_PATHEFFECTS,
            )
            if track == len(split_idx) - 2:
                ax.axvline(
                    b - 0.5,
                    color=TRACK_BOUNDARY_COLOR,
                    linestyle="--",
                    path_effects=TRACK_BOUNDARY_PATHEFFECTS,
                )
            ax.annotate(
                f"{track+1}",
                ((a + b) / 2, 1),
                color=TRACK_BOUNDARY_COLOR,
                path_effects=TRACK_BOUNDARY_PATHEFFECTS,
                horizontalalignment="center",
            )

    if ignored_cols is not None:
        for c in ignored_cols.nonzero(as_tuple=True)[1]:
            ax.fill_betweenx(
                (-0.5, W.shape[0] - 0.5),
                c - 0.5,
                c,
                color=IGNORED_COLOR,
                alpha=IGNORED_ALPHA,
            )

    return im


def plot_nmf(learner, internal=False):
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))

    im = plot_pow_spec(
        learner.nmf.V.cpu().detach().numpy()
        if internal
        else learner.V.cpu().detach().numpy(),
        ax=axes[0, 0],
    )
    fig.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title("V")

    im = plot_H(
        learner.nmf.H.cpu().detach().numpy()
        if internal
        else learner.H.cpu().detach().numpy(),
        split_idx=learner.split_idx,
        ignored_lines=learner.W_ignored_cols,
        ax=axes[0, 1],
    )
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title("H")

    im = plot_pow_spec(
        learner.nmf.W.cpu().detach().numpy()
        if internal
        else learner.W.cpu().detach().numpy(),
        split_idx=learner.split_idx,
        ignored_cols=learner.W_ignored_cols,
        ax=axes[1, 0],
    )
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title("W")

    im = plot_pow_spec(
        (learner.W @ learner.H).cpu().detach().numpy()
        if internal
        else (learner.nmf.W @ learner.nmf.H).cpu().detach().numpy(),
        ax=axes[1, 1],
    )
    fig.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title("WH")

    plt.tight_layout()
    return fig
