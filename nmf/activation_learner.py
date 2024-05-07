import numpy as np
from typing import List, Callable, Optional
import warnings
from beta_nmf import BetaNMF
import scipy.ndimage
import scipy.signal
import librosa
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ActivationLearner:
    def __init__(
        self,
        mix: np.ndarray,
        refs: List[np.ndarray],
        fs,
        additional_dim: int = 0,
        col_mag_threshold=1e-8,
        win_len=2**14,
        hop_len=2**12,
        n_mels=512,
        **nmf_kwargs,
    ):
        self.learn_add = additional_dim > 0
        self.refs = refs
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.fs = fs
        logger.info(f"{win_len=}={win_len/fs:.2f}s")
        logger.info(f"{hop_len=}={hop_len/fs:.2f}s")

        # transform=lambda x: abs(librosa.stft(x, n_fft=NFFT, hop_length=HLEN, center=True))**2,
        # inv_transform=lambda S: librosa.istft(S, n_fft=NFFT, hop_length=HLEN, center=True),

        # transform = lambda x: abs(librosa.cqt(x, sr=FS, hop_length=HLEN)),
        # inv_transform=lambda S: librosa.icqt(S, sr=FS, hop_length=HLEN),

        # transform = lambda x: abs(librosa.feature.mfcc(y=x, sr=FS)),
        # inv_transform=lambda mfcc: librosa.feature.inverse.mfcc_to_audio(mfcc),

        self.transform = lambda x: librosa.feature.melspectrogram(
            y=x, sr=fs, n_fft=win_len, hop_length=hop_len, power=2, n_mels=n_mels
        )
        self.inv_transform = lambda S: librosa.feature.inverse.mel_to_audio(
            S, sr=fs, n_fft=win_len, hop_length=hop_len, power=2
        )

        # transform=lambda x: beat_stft(x, FS, NFFT)[0],
        # inv_transform=lambda S: librosa.istft(S, n_fft=NFFT, hop_length=HLEN, center=True),

        # transform and clean audio into feature matrix
        mix_mat = self.transform(mix)
        refs_mat = []
        for i in refs:
            spec = self.transform(i)
            spec /= spec.max()
            # set all nearzero columns to zero
            spec[:, np.mean(spec, axis=0) < col_mag_threshold] = 0
            refs_mat.append(spec)

        # compute indexes of track boundaries
        self.split_idx = [0] + list(
            np.cumsum([ref.shape[1] for ref in refs_mat], axis=0)
        )

        # construct NMF matrices
        V = mix_mat / mix_mat.max()

        refs_mat = [i / i.max() for i in refs_mat]
        if self.learn_add:
            Wa = np.random.rand(refs_mat[0].shape[0], additional_dim)
            W = np.concatenate(refs_mat + [Wa], axis=1)
            self.split_idx.append(self.split_idx[-1] + additional_dim)
        else:
            W = np.concatenate(refs_mat, axis=1)

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])

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
        self.nmf.H = np.clip(self.nmf.H, 0, 1)

        # Calculate loss
        loss = self.nmf.loss()
        assert not np.isnan(loss).any(), "NaN in loss"

        return loss

    def volume(self, weiner: Optional[int] = 3, medfilt: Optional[int] = 7):
        ret = []
        if weiner is not None:
            Hfilt = scipy.signal.wiener(self.nmf.H, mysize=(weiner, weiner))
        else:
            Hfilt = self.nmf.H

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
        if weiner is not None:
            Hfilt = scipy.signal.wiener(self.nmf.H, mysize=(weiner, weiner))
        else:
            Hfilt = self.nmf.H
            
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
        if i < len(self.refs):
            ref_spec = self.transform(self.refs[i])
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
        SPECFUNC = lambda S, ax: librosa.display.specshow(
            librosa.power_to_db(S),
            ax=ax,
            cmap=CMAP,
            hop_length=self.hop_len,
            sr=self.fs,
            x_axis="time",
            y_axis="mel",
        )

        # Plot the H matrix
        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        im = axes[0, 0].imshow(self.nmf.H, cmap=CMAP, aspect="auto", origin="lower")
        axes[0, 0].set_title("H (activations)")
        axes[0, 0].set_xlabel("mix frame")
        axes[0, 0].set_ylabel("ref frame")
        fig.colorbar(im, ax=axes[0, 0])

        im = SPECFUNC(self.nmf.W, axes[0, 1])
        fig.colorbar(mappable=im, ax=axes[0, 1])
        axes[0, 1].set_title("$W$ (reference tracks)")

        for track, (a, b) in enumerate(zip(self.split_idx, self.split_idx[1:])):
            axes[0, 0].axhline(a, color="r", linestyle="--")
            axes[0, 0].annotate(f"track {track}", (0, (a + b) / 2), color="red")
            axes[0, 1].axvline(a / self.fs * self.hop_len, color="r", linestyle="--")
            axes[0, 1].annotate(
                f"track {track}", ((a + b) / 2 / self.fs * self.hop_len, 1), color="red"
            )

        im = SPECFUNC(self.nmf.V, axes[1, 0])
        axes[1, 0].set_title("$V$ (mix)")
        fig.colorbar(mappable=im, ax=axes[1, 0])

        im = SPECFUNC(self.nmf.W @ self.nmf.H, axes[1, 1])
        axes[1, 1].set_title("$\\hat{V} = WH$ (estimated mix)")
        fig.colorbar(mappable=im, ax=axes[1, 1])

        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
        for i, v in enumerate(
            iterable=self.position(threshold=5e-3, weiner=None, medfilt=None)
        ):
            axes[2, 0].plot(v, alpha=0.2, color=colors[i % len(colors)])
        for i, v in enumerate(iterable=self.position(threshold=5e-3)):
            axes[2, 0].plot(v, label=f"track {i}", color=colors[i % len(colors)])
        axes[2, 0].legend()
        axes[2, 0].set_title("track time")
        axes[2, 0].set_xlabel("mix frame")
        axes[2, 0].set_ylabel("ref frame")

        for i, v in enumerate(iterable=self.volume(weiner=None, medfilt=None)):
            axes[2, 1].plot(v, alpha=0.2, color=colors[i % len(colors)])
        for i, v in enumerate(self.volume()):
            axes[2, 1].plot(v, label=f"track {i}", color=colors[i % len(colors)])
        axes[2, 1].legend()
        axes[2, 1].set_title("track volume")
        axes[2, 1].set_xlabel("mix frame")

        plt.tight_layout()
        plt.show()
