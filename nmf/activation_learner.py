import numpy as np
import warnings
import enum
import scipy.sparse
from beta_nmf import BetaNMF
import scipy.ndimage
import scipy.signal
import librosa
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import functools

logger = logging.getLogger(__name__)


def _transform_melspec(input, fs, n_mels, stft_win_func, win_len, hop_len):
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

    return melspec, spec


class ActivationLearner:
    def __init__(
        self,
        inputs: list[np.ndarray],
        fs: int,
        win_size: float,
        hop_size: float,
        beta=0,
        additional_dim: int = 0,
        stft_win_func: str = "hann",
        n_mels: int = 512,
        polyphony_penalty: float = 0,
        **nmf_kwargs,
    ):
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
        self.stft_win_func = stft_win_func

        # transform inputs
        with multiprocessing.Pool() as pool:
            f = functools.partial(
                _transform_melspec,
                fs=fs,
                n_mels=n_mels,
                stft_win_func=stft_win_func,
                win_len=win_len,
                hop_len=hop_len,
            )
            input_powspecs, input_specs = zip(
                *tqdm(
                    pool.imap(f, inputs),
                    desc="Transforming inputs",
                    total=len(inputs),
                )
            )

        # save specs for reconstruction
        self.input_specs = input_specs

        # compute indexes of track boundaries
        self.split_idx = [0] + list(
            np.cumsum([ref.shape[1] for ref in input_powspecs[:-1]], axis=0)
        )

        # construct NMF matrices
        V = input_powspecs[-1]

        if self.learn_add:
            Wa = np.random.rand(input_powspecs[0].shape[0], additional_dim)
            W = np.concatenate(input_powspecs[:-1] + [Wa], axis=1)
            self.split_idx.append(self.split_idx[-1] + additional_dim)
        else:
            W = np.concatenate(input_powspecs[:-1], axis=1)

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])
        H = scipy.sparse.bsr_array(H)

        logger.debug(f"Shape of W: {W.shape}")
        logger.debug(f"Shape of H: {H.shape}")
        logger.debug(f"Shape of V: {V.shape}")

        self.nmf = BetaNMF(V, W, H, beta, fixed_W=not self.learn_add, **nmf_kwargs)

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
            Vi = self.nmf.V * (self.input_specs @ self.H[a:b, :]) / (self.W @ self.H)
        else:  # no phase :(
            raise NotImplementedError("Cannot reconstruct without original material")
            # warnings.warn(f"Track {i} not in refs: i don't have phase info")
            # Vi = (
            # self.nmf.V
            # * (self.nmf.W[:, a:b] @ self.nmf.H[a:b, :])
            # / (self.nmf.W @ self.nmf.H)
            # )

        audio = librosa.istft(
            Vi,
            n_fft=int(self.fs * self.win_size),
            hop_length=int(self.fs * self.hop_size),
            win_length=int(self.fs * self.win_size),
            center=False,
            window=self.stft_win_func,
        )
        return audio

    @property
    def H(self):
        if isinstance(self.nmf.H, scipy.sparse.sparray):
            return self.nmf.H.toarray()
        else:
            return self.nmf.H

    @property
    def W(self):
        return self.nmf.W

    @property
    def V(self):
        return self.nmf.V
