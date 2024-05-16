import numpy as np
from typing import List, Callable, Optional
import warnings

import scipy.sparse
from beta_nmf import BetaNMF
import scipy.ndimage
import scipy.signal
import librosa
import logging
import matplotlib.pyplot as plt
import beat_track
from tqdm import tqdm
import multiprocessing
import functools

logger = logging.getLogger(__name__)


def transform(input, fs, n_mels, stft_win_func, spec_conv_win):
    x, bounds = input
    stft, n_fft = beat_track.variable_stft(x, bounds, win_func=stft_win_func)
    if spec_conv_win is not None:
        stft = scipy.ndimage.convolve1d(stft, spec_conv_win, axis=1, mode="constant")

    mel_f = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)
    melspec = mel_f.dot(abs(stft) ** 2)
    melspec /= np.sum(melspec, axis=0, keepdims=True)  # normalize columns

    return melspec


class ActivationLearner:
    def __init__(
        self,
        inputs: list[np.ndarray],
        fs: int,
        additional_dim: int = 0,
        boundaries: list[np.ndarray] = None,
        stft_win_func: Callable = None,
        spec_conv_win: np.ndarray = None,
        win_size: float = None,
        overlap: float = None,
        n_mels: int = 512,
        **nmf_kwargs,
    ):
        assert (boundaries is not None) or (
            win_size is not None and overlap is not None
        )

        if boundaries is None:
            assert 0 <= overlap <= 1
            win_len = int(win_size * fs)
            hop_len = int(win_size * fs * (1 - overlap))
            logger.info(f"{win_len=}")
            logger.info(f"{hop_len=}")
            boundaries = []
            for i in inputs:
                assert win_len < len(i), "window size cannot be larger than input"
                start = np.arange(0, len(i) - win_len, hop_len)
                end = start + win_len
                boundaries.append(np.vstack([start, end]).T)

        self.learn_add = additional_dim > 0
        self.inputs = inputs
        self.boundaries = boundaries
        self.n_mels = n_mels
        self.fs = fs

        # transform inputs
        inputs_mat: list[np.ndarray] = []
        with multiprocessing.Pool() as pool:
            f = functools.partial(transform, fs=fs, n_mels=n_mels, stft_win_func=stft_win_func, spec_conv_win=spec_conv_win)
            inputs_mat: list[np.ndarray] = list(
                tqdm(
                    pool.imap(f, zip(inputs, boundaries)),
                    desc="Transforming inputs",
                    total=len(inputs),
                )
            )
            

        # compute indexes of track boundaries
        self.split_idx = [0] + list(
            np.cumsum([ref.shape[1] for ref in inputs_mat[:-1]], axis=0)
        )

        # construct NMF matrices
        V = inputs_mat[-1]

        if self.learn_add:
            Wa = np.random.rand(inputs_mat[0].shape[0], additional_dim)
            W = np.concatenate(inputs_mat[:-1] + [Wa], axis=1)
            self.split_idx.append(self.split_idx[-1] + additional_dim)
        else:
            W = np.concatenate(inputs_mat[:-1], axis=1)

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])
        H = scipy.sparse.bsr_array(H)

        logger.debug(f"Shape of W: {W.shape}")
        logger.debug(f"Shape of H: {H.shape}")
        logger.debug(f"Shape of V: {V.shape}")

        self.nmf = BetaNMF(V, W, H, 0, fixed_W=not self.learn_add, **nmf_kwargs)

    def iterate(self):
        if self.learn_add:
            # save everything except Wa
            W_save = self.nmf.W[:, : self.split_idx[-2]].copy()
            self.nmf.iterate()
            # copy it back
            self.nmf.W[:, : self.split_idx[-2]] = W_save
            # clip Wa
            self.nmf.W[:, self.split_idx[-2] : self.split_idx[-1]] = np.clip(
                self.nmf.W[:, self.split_idx[-2] : self.split_idx[-1]], 0, 1
            )
        else:
            self.nmf.iterate()
        H = self.nmf.H.toarray()
        H = np.clip(H, 0, 1)
        self.nmf.H = scipy.sparse.bsr_array(H)

        # Calculate loss
        loss = self.nmf.loss()
        assert not np.isnan(loss).any(), "NaN in loss"

        return loss

    def volume(self, weiner: Optional[int] = 3, medfilt: Optional[int] = 7):
        ret = []
        H = self.nmf.H.toarray()
        if weiner is not None:
            Hfilt = scipy.signal.wiener(H, mysize=(weiner, weiner))
        else:
            Hfilt = H

        sum = Hfilt.sum(axis=0)
        for left, right in zip(self.split_idx, self.split_idx[1:]):
            vol = Hfilt[left:right, :].sum(axis=0) / sum
            if medfilt is not None:
                vol = scipy.signal.medfilt(vol, medfilt)
            ret.append(vol)
        return ret

    def position(
        self, threshold=1e-5, weiner: Optional[int] = 3, medfilt: Optional[int] = 7
    ):
        ret = []
        H = self.nmf.H.toarray()
        if weiner is not None:
            Hfilt = scipy.signal.wiener(H, mysize=(weiner, weiner))
        else:
            Hfilt = H

        for left, right in zip(self.split_idx, self.split_idx[1:]):
            _, N = Hfilt.shape
            pos = np.empty(N)
            for i in range(N):
                col = Hfilt[left:right, i] ** 2
                if col.sum() >= threshold:
                    pos[i] = scipy.ndimage.center_of_mass(col)[0]
                else:
                    pos[i] = np.nan
            if medfilt is not None:
                pos = scipy.signal.medfilt(pos, medfilt)
            ret.append(pos)
        return ret

    def reconstruct(self, i: int):
        a = self.split_idx[i]
        b = self.split_idx[i + 1]
        if i < len(self.inputs) - 1:
            ref_spec = self.transform(self.inputs[i])
            Vi = (
                self.nmf.V * (ref_spec @ self.nmf.H[a:b, :]) / (self.nmf.W @ self.nmf.H)
            )
        else:  # no phase :(
            Vi = (
                self.nmf.V
                * (self.nmf.W[:, a:b] @ self.nmf.H[a:b, :])
                / (self.nmf.W @ self.nmf.H)
            )
            warnings.warn(f"Track {i} not in refs: i don't have phase info")
        return self.inv_transform(Vi)

    def plot(self):
        CMAP = "turbo"
        SPECFUNC = lambda S, ax: ax.imshow(
            librosa.power_to_db(S),
            cmap=CMAP,
            aspect="auto",
            origin="lower",
            interpolation="none",
        )

        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        # plot H
        im = axes[0, 0].imshow(
            self.nmf.H.toarray(), cmap=CMAP, aspect="auto", origin="lower"
        )
        axes[0, 0].set_title("H (activations)")
        axes[0, 0].set_xlabel("mix frame")
        axes[0, 0].set_ylabel("ref frame")
        fig.colorbar(im, ax=axes[0, 0])

        # plot W
        im = SPECFUNC(self.nmf.W, axes[0, 1])
        fig.colorbar(mappable=im, ax=axes[0, 1])
        axes[0, 1].set_title("$W$ (reference tracks)")

        # annotate track boundaries
        for track, (a, b) in enumerate(zip(self.split_idx, self.split_idx[1:])):
            axes[0, 0].axhline(a - 0.5, color="r", linestyle="--")
            axes[0, 0].annotate(f"track {track}", (0, (a + b) / 2), color="red")
            axes[0, 1].axvline(a - 0.5, color="r", linestyle="--")
            axes[0, 1].annotate(f"track {track}", ((a + b) / 2, 1), color="red")

        im = SPECFUNC(self.nmf.V, axes[1, 0])
        axes[1, 0].set_title("$V$ (mix)")
        fig.colorbar(mappable=im, ax=axes[1, 0])

        im = SPECFUNC(self.nmf.W @ self.nmf.H, axes[1, 1])
        axes[1, 1].set_title("$\\hat{V} = WH$ (estimated mix)")
        fig.colorbar(mappable=im, ax=axes[1, 1])

        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
        positions = self.position(threshold=0, weiner=None, medfilt=None)
        volumes = self.volume(weiner=None, medfilt=None)

        for i, v in enumerate(positions):
            v = np.vstack([np.arange(len(v)), v]).T
            for j, ((x0, y0), (x1, y1)) in enumerate(zip(v[:-1], v[1:])):
                axes[2, 0].plot(
                    (x0, x1),
                    (y0, y1),
                    color=colors[i % len(colors)],
                    alpha=volumes[i][j] ** 2,
                )
        axes[2, 0].set_title("track time")
        axes[2, 0].set_xlabel("mix frame")
        axes[2, 0].set_ylabel("ref frame")

        for i, v in enumerate(volumes):
            axes[2, 1].plot(
                v,
                color=colors[i % len(colors)],
                label=f"track {i}",
            )
        axes[2, 1].legend()
        axes[2, 1].set_title("track volume")
        axes[2, 1].set_xlabel("mix frame")

        plt.tight_layout()
        plt.show()
