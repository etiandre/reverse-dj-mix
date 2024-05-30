import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def volumes(H: np.ndarray, colsum, split_idx):
    H = H.copy()
    # de-normalize by mix
    # H *= colsum[-1]
    # de-normalize by each track
    # H /= np.hstack(colsum[:-1]).T

    volumes = []
    for left, right in zip(split_idx, split_idx[1:]):
        H_track = H[left:right, :]
        vol = H_track.sum(axis=0) / H.sum(axis=0)  # TODO: test other statistics ?
        # vol = np.sqrt(H_track.sum(axis=0))  # TODO: test other statistics ?
        volumes.append(vol)
    return volumes


def positions(H: np.ndarray, split_idx, hop_size):
    ret = []
    for left, right in zip(split_idx, split_idx[1:]):
        _, N = H.shape
        pos = np.empty(N)
        for i in range(N):
            col = H[left:right, i] ** 2  # TODO: not rigoureux
            pos[i] = scipy.ndimage.center_of_mass(col)[0] * hop_size
        ret.append(pos)
    return ret


def plot_vol_pos(
    volumes: np.ndarray,
    positions: np.ndarray,
    hop_size: float,
    real_volumes: np.ndarray = None,
    real_positions: np.ndarray = None,
):
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    fig, axes = plt.subplots(2, 1, figsize=(5 * len(volumes), 9))

    def plot_pos(positions, real):
        for i, ref_time in enumerate(positions):
            mix_time = np.arange(len(ref_time)) * hop_size
            coords = np.vstack([mix_time, ref_time]).T
            for j, ((x0, y0), (x1, y1)) in enumerate(zip(coords[:-1], coords[1:])):
                axes[0].plot(
                    (x0, x1),
                    (y0, y1),
                    "--" if real else "-",
                    color=colors[i % len(colors)],
                    # alpha=np.clip(volumes[i][j]**2, 0, 1),
                )

    plot_pos(positions, False)
    if real_positions is not None:
        plot_pos(real_positions, True)
    axes[0].set_title("track time")
    axes[0].set_xlabel("mix time (s)")
    axes[0].set_ylabel("ref time (s)")
    # axes[0].set_aspect("equal")

    def plot_vol(volumes, real):
        for i, v in enumerate(volumes):
            mix_time = np.arange(len(v)) * hop_size
            axes[1].plot(
                mix_time,
                v,
                "--" if real else "-",
                color=colors[i % len(colors)],
                label=f"track {i}" if real else None,
            )

    plot_vol(volumes, False)
    if real_volumes is not None:
        plot_vol(real_volumes, True)
    axes[1].legend()
    axes[1].set_title("track volume")
    axes[1].set_xlabel("mix time (s)")
    axes[1].set_ylim(0,1)

    plt.tight_layout()
    plt.show()
