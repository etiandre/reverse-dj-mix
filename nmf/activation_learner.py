import abc
from typing import Sequence, Union
import numpy as np
import librosa
import logging

from tqdm import tqdm
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


class Warmup(abc.ABC):
    @abc.abstractmethod
    def get(self, iter: int) -> float:
        pass


class LinearWarmup(Warmup):
    def __init__(self, p0: float, p1: float, i0: int, i1: int):
        self.p0 = p0
        self.p1 = p1
        self.i0 = i0
        self.i1 = i1

    def get(self, iter: int):
        if iter < self.i0:
            return self.p0
        elif iter > self.i1:
            return self.p1
        else:
            return self.p0 + (self.p1 - self.p0) * (iter - self.i0) / (
                self.i1 - self.i0
            )


class ActivationLearner:
    def __init__(
        self,
        inputs: list[np.ndarray],
        fs: int,
        win_size: float,
        hop_size: float,
        divergence: Divergence,
        penalties: Sequence[tuple[Penalty, Union[float, Warmup]]],
        stft_win_func: str = "hann",
        n_mels: int = 512,
        noise_floor: float = 0.01,
        use_gpu: bool = False,
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
        W += noise_floor
        # normalize W and V
        self.W_norm_factor = W.sum(axis=0, keepdims=True)
        # self.W_norm_factor[self.W_norm_factor == 0] = 1
        assert not np.any(self.W_norm_factor == 0)
        self.V_norm_factor = V.sum()
        W = W / self.W_norm_factor
        V = V / self.V_norm_factor

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])

        logger.info(f"Shape of W: {W.shape}")
        logger.info(f"Shape of H: {H.shape}")
        logger.info(f"Shape of V: {V.shape}")

        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        pen_func, self.pen_warmups = zip(*penalties)
        self.nmf = NMF(
            torch.Tensor(V).to(device),
            torch.Tensor(W).to(device),
            torch.Tensor(H).to(device),
            divergence,
            [],
            pen_func,
            trainable_W=False,
        ).to(device)

    def fit(self, iter_max: int, loss_every: int = 50):
        logger.info(
            f"Running NMF on V:{self.nmf.V.shape}, W:{self.nmf.W.shape}, H:{self.nmf.H.shape}"
        )
        loss_history = []
        loss = np.inf
        for i in tqdm(range(iter_max)):
            pen_lambdas = [
                w.get(i) if isinstance(w, Warmup) else w for w in self.pen_warmups
            ]
            self.nmf.iterate([], pen_lambdas)
            if i % loss_every == 0:
                loss, loss_components = self.loss(pen_lambdas)
                loss_history.append(loss_components)

            if i >= iter_max:
                logger.info(f"Stopped at NMF iteration={i} loss={loss}")
                break
        return loss_history

    def loss(self, pen_lambdas: list[float]):
        full_loss, losses = self.nmf.loss([], pen_lambdas)
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
        return (
            self.nmf.H.cpu().detach().numpy()
            * self.V_norm_factor
            / self.W_norm_factor.T
        )

    @property
    def W(self):
        return self.nmf.W.cpu().detach().numpy() * self.W_norm_factor

    @property
    def V(self):
        return self.nmf.V.cpu().detach().numpy() * self.V_norm_factor
