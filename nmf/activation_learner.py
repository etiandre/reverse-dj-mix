import numpy as np
import librosa
import logging
from pytorch_nmf import Divergence, Penalty, NMF
import torch

logger = logging.getLogger(__name__)


def _transform_melspec(input, fs, n_mels, stft_win_func, win_len, hop_len):
    spec = librosa.stft(
        input,
        n_fft=win_len,
        hop_length=hop_len,
        win_length=win_len,
        center=True,
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
        divergence: Divergence,
        penalties: list[tuple[Penalty, float]],
        stft_win_func: str = "hann",
        n_mels: int = 512,
        low_power_factor: float = 0.01,
    ):
        win_len = int(win_size * fs)
        hop_len = int(hop_size * fs)
        logger.info(f"{win_len=}")
        logger.info(f"{hop_len=}")
        logger.info(f"overlap={1-hop_size/win_size:%}")

        self.inputs = inputs
        self.n_mels = n_mels
        self.fs = fs
        self.win_size = win_size
        self.hop_size = hop_size
        self.stft_win_func = stft_win_func

        # transform inputs
        logger.info("Transforming inputs")
        input_powspecs, input_specs = zip(
            *[
                _transform_melspec(
                    i,
                    fs=fs,
                    n_mels=n_mels,
                    stft_win_func=stft_win_func,
                    win_len=win_len,
                    hop_len=hop_len,
                )
                for i in inputs
            ]
        )

        # save specs for reconstruction
        self.input_specs = input_specs

        # compute indexes of track boundaries
        self.split_idx = [0] + list(
            np.cumsum([ref.shape[1] for ref in input_powspecs[:-1]], axis=0)
        )

        # construct NMF matrices
        V = input_powspecs[-1]

        W = np.concatenate(input_powspecs[:-1], axis=1)

        # fill the columns of W with too little power with noise to prevent explosion in NMF
        # frames_power = W.sum(axis=0)
        # low_frames = frames_power / np.max(frames_power) < low_power_factor
        # print(frames_power / np.max(frames_power))
        # if np.sum(low_frames) > 0:
        #     logger.info(f"Filling {np.sum(low_frames)} low frames with noise")
        # W[:, low_frames] = np.random.rand(W.shape[0], np.sum(low_frames))
        W += np.abs(np.random.randn(*W.shape)) * low_power_factor
        # normalize W and V
        self.W_norm_factor = W.sum(axis=0, keepdims=True)
        assert not np.any(self.W_norm_factor == 0)
        self.V_norm_factor = V.sum()
        W = W / self.W_norm_factor
        V = V / self.V_norm_factor

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])

        logger.info(f"Shape of W: {W.shape}")
        logger.info(f"Shape of H: {H.shape}")
        logger.info(f"Shape of V: {V.shape}")

        self.nmf = NMF(
            torch.Tensor(V),
            torch.Tensor(W),
            torch.Tensor(H),
            divergence,
            [],
            penalties,
            trainable_W=False,
        )

    def iterate(self):
        self.nmf.iterate()

    def loss(self):
        full_loss, losses = self.nmf.loss()
        losses["penalties"] = losses["penalties_H"]
        del losses["penalties_W"]
        return full_loss, losses

    def reconstruct_tracks(self):
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
                center=True,
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
            center=True,
            window=self.stft_win_func,
        )
        return audio

    @property
    def H(self):
        return self.nmf.H.detach().numpy() * self.V_norm_factor / self.W_norm_factor.T

    @property
    def W(self):
        return self.nmf.W.detach().numpy() * self.W_norm_factor

    @property
    def V(self):
        return self.nmf.V.detach().numpy() * self.V_norm_factor
