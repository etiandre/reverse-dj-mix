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
        vol = H_track.sum(axis=0)/H.sum(axis=0)  # TODO: test other statistics ?
        volumes.append(vol)
    return volumes


def positions(H: np.ndarray, split_idx):
    ret = []
    for left, right in zip(split_idx, split_idx[1:]):
        _, N = H.shape
        pos = np.empty(N)
        for i in range(N):
            col = H[left:right, i] ** 2  # TODO: not rigoureux
            pos[i] = scipy.ndimage.center_of_mass(col)[0]
        ret.append(pos)
    return ret


def frames_to_times(frames, bounds, fs, center=False):
    if center:
        times = np.mean(bounds, axis=1) / fs
    else:
        times = bounds[:, 0] / fs

    return np.interp(frames, np.arange(len(times)), times)


def plot_vol_pos(
    volumes: np.ndarray, positions: np.ndarray, bounds: np.ndarray, fs, center=True
):
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    mix_time = frames_to_times(np.arange(len(bounds[-1])), bounds[-1], fs, center)

    fig, axes = plt.subplots(2, 1, figsize=(5 * len(volumes), 6))
    for i, ref_frames in enumerate(positions):
        y = frames_to_times(ref_frames, bounds[i], fs, center)
        coords = np.vstack([mix_time, y]).T
        for j, ((x0, y0), (x1, y1)) in enumerate(zip(coords[:-1], coords[1:])):
            axes[0].plot(
                (x0, x1),
                (y0, y1),
                "--" if volumes[i][j] < 0.1 else "-",
                color=colors[i % len(colors)],
                alpha=np.clip(0.5 + volumes[i][j] / 2, 0, 1),
            )
    axes[0].set_title("track time")
    axes[0].set_xlabel("mix time (s)")
    axes[0].set_ylabel("ref time (s)")
    axes[0].set_aspect("equal")

    for i, v in enumerate(volumes):
        axes[1].plot(
            mix_time,
            v,
            color=colors[i % len(colors)],
            label=f"track {i}",
        )
    axes[1].legend()
    axes[1].set_title("track volume")
    axes[1].set_xlabel("mix time (s)")

    plt.tight_layout()
    plt.show()
