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


ENERGY_MIN = 1e-1


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

        # remove columns of W with too little energy to prevent explosion in NMF
        W[:, W.mean(axis=0) < ENERGY_MIN] = 0
        
        # normalize W and V
        self.W_norm_factor = W.sum(axis=0, keepdims=True)
        self.W_norm_factor[self.W_norm_factor == 0] = 1
        self.V_norm_factor = V.sum()
        W = W / self.W_norm_factor
        V = V / self.V_norm_factor

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])
        H = scipy.sparse.bsr_array(H)

        logger.info(f"Shape of W: {W.shape}")
        logger.info(f"Shape of H: {H.shape}")
        logger.info(f"Shape of V: {V.shape}")

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

        if self.polyphony_penalty > 0:
            H = self.nmf.H.toarray()
            H_ = H.copy()
            poly_limit = 1  # maximum simultaneous activations in one column
            colCutoff = -np.partition(-H, poly_limit, 0)[poly_limit, :]
            H_[H_ < colCutoff[None, :]] *= 1 - self.polyphony_penalty
            H = (1 - regulation_strength) * H + regulation_strength * H_
            self.nmf.H = scipy.sparse.bsr_array(H)

        # TODO: clip efficiently
        # self.nmf.H[self.nmf.H > 1e3] = 1e3

        # Calculate loss
        loss = self.nmf.loss()
        assert not np.isnan(loss).any(), "NaN in loss"

        return loss

    def reconstruct_tracks(self):
        # TODO: does not work with such big nfft (need too much memory for inverse mel)
        ret = []
        mix_spec = self.input_specs[-1]
        W_spec = np.concatenate(self.input_specs[:-1], axis=1)
        for i, (a, b) in enumerate(zip(self.split_idx[:-1], self.split_idx[1:])):
            track_spec = self.input_specs[i]
            Vi = mix_spec * (track_spec @ self.H[a:b, :]) / (W_spec @ self.H)
            audio = librosa.istft(
                Vi,
                n_fft=int(self.fs * self.win_size),
                hop_length=int(self.fs * self.hop_size),
                win_length=int(self.fs * self.win_size),
                center=False,
                window=self.stft_win_func,
            )
            ret.append(audio)
        return ret

    def reconstruct_mix(self):
        W_spec = np.concatenate(self.input_specs[:-1], axis=1)
        Vhat = W_spec @ self.H
        audio = librosa.istft(
            Vhat,
            n_fft=int(self.fs * self.win_size),
            hop_length=int(self.fs * self.hop_size),
            win_length=int(self.fs * self.win_size),
            center=False,
            window=self.stft_win_func,
        )
        return audio

    @property
    def H(self):
        return self.nmf.H.toarray() * self.V_norm_factor / self.W_norm_factor.T

    @property
    def W(self):
        return self.nmf.W * self.W_norm_factor

    @property
    def V(self):
        return self.nmf.V * self.V_norm_factor
