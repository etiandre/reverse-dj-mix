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


def transform_melspec(input, fs, n_mels, stft_win_func, win_len, hop_len):
    spec = librosa.stft(
        input,
        n_fft=win_len,
        hop_length=hop_len,
        win_length=win_len,
        center=False,
        window=stft_win_func,
    )
    mel_f = librosa.filters.mel(sr=fs, n_fft=win_len, n_mels=n_mels)
    melspec = mel_f.dot(abs(spec) ** 2)
    colsum = np.sum(melspec, axis=0, keepdims=True)
    colsum[colsum == 0] = 1e-30  # TODO this is a hack
    melspec /= colsum  # normalize columns

    return melspec, colsum


class ActivationLearner:
    def __init__(
        self,
        inputs: list[np.ndarray],
        fs: int,
        win_size: float,
        hop_size: float,
        additional_dim: int = 0,
        stft_win_func: str = "hann",
        n_mels: int = 512,
        polyphony_penalty: float = 0,
        **nmf_kwargs,
    ):
        assert hop_size < win_size
        win_len = int(win_size * fs)
        hop_len = int(hop_size * fs)
        logger.info(f"{win_len=}")
        logger.info(f"{hop_len=}")
        logger.info(f"overlap={1-hop_size/win_size:%}")

        self.learn_add = additional_dim > 0
        self.inputs = inputs
        self.n_mels = n_mels
        self.fs = fs
        self.polyphony_penalty = polyphony_penalty
        self.win_size = win_size
        self.hop_size = hop_size

        # transform inputs
        with multiprocessing.Pool() as pool:
            f = functools.partial(
                transform_melspec,
                fs=fs,
                n_mels=n_mels,
                stft_win_func=stft_win_func,
                win_len=win_len,
                hop_len=hop_len,
            )
            out = list(
                tqdm(
                    pool.imap(f, inputs),
                    desc="Transforming inputs",
                    total=len(inputs),
                )
            )
            inputs_mat, colsum = zip(*out)
        self.colsum = colsum

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

    def iterate(self, regulation_strength: float = 1.0):
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

        if self.polyphony_penalty > 0:
            H_ = H.copy()
            poly_limit = 1  # maximum simultaneous activations in one column
            colCutoff = -np.partition(-H, poly_limit, 0)[poly_limit, :]
            H_[H_ < colCutoff[None, :]] *= 1 - self.polyphony_penalty
            H = (1 - regulation_strength) * H + regulation_strength * H_

        H = np.clip(H, 0, 1)

        self.nmf.H = scipy.sparse.bsr_array(H)

        # Calculate loss
        loss = self.nmf.loss()
        assert not np.isnan(loss).any(), "NaN in loss"

        return loss

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

        fig, axes = plt.subplots(2, 2, figsize=(15, 6))

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

        plt.tight_layout()
        plt.show()
