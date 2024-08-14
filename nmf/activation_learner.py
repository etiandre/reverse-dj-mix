import abc
from typing import Optional, Sequence, Union
import numpy as np
import librosa
import logging

from tqdm import tqdm
from pytorch_nmf import EPS, Divergence, Penalty, NMF
import torch
import carve
import plot
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _transform_melspec(input, fs, n_mels, stft_win_func, win_len, hop_len, power):
    spec = librosa.stft(
        input,
        n_fft=win_len,
        hop_length=hop_len,
        win_length=win_len,
        center=True,
        window=stft_win_func,
    )
    mel_f = librosa.filters.mel(sr=fs, n_fft=win_len, n_mels=n_mels)
    melspec = mel_f.dot(abs(spec) ** power)

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
        spec_power: float,
        stft_win_func: str = "hann",
        n_mels: int = 512,
        low_power_threshold: float = 0.01,
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
                    power=spec_power,
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

        # normalize W and V
        W_col_power = W.sum(axis=0, keepdims=True)
        W_ignored_cols = W_col_power / W.shape[0] < low_power_threshold
        if np.any(W_ignored_cols):
            logger.warning(f"Ignored columns: {np.where(W_ignored_cols)[1]}")

        W_col_power[W_ignored_cols] = 1
        V_norm_fac = V.sum()
        W = W / W_col_power
        V = V / V_norm_fac

        self.W_norm_fac = torch.Tensor(W_col_power)
        self.V_norm_fac = V_norm_fac
        self.W_ignored_cols = torch.from_numpy(W_ignored_cols)

        # initialize activation matrix
        # H = np.random.rand(W.shape[1], V.shape[1])
        H = np.ones((W.shape[1], V.shape[1]))
        H[W_ignored_cols.flatten(), :] = 0

        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        if len(penalties) == 0:
            pen_func, self.pen_warmups = [], []
        else:
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

    def fit(
        self, iter_max: int, loss_every: int = 50, dloss_min: Optional[float] = None
    ):
        logger.info(
            f"Running NMF on V:{self.nmf.V.shape}, W:{self.nmf.W.shape}, H:{self.nmf.H.shape}"
        )
        loss_history = []
        loss = np.inf
        last_loss = np.inf
        for i in (pbar := tqdm(range(iter_max))):
            pen_lambdas = [
                w.get(i) if isinstance(w, Warmup) else w for w in self.pen_warmups
            ]
            self.nmf.iterate([], pen_lambdas)
            with torch.no_grad():
                self.nmf.H.clamp_(max=torch.Tensor(self.W_norm_fac.T / self.V_norm_fac))

            if i % loss_every == 0:
                loss, loss_components = self.loss(pen_lambdas)
                dloss = (last_loss - loss) / loss_every
                pbar.set_description(f"Loss={loss:.2e}, dLoss = {dloss:.2e}")
                last_loss = loss
                loss_history.append(loss_components)

                if dloss_min is not None and dloss <= dloss_min:
                    break

            if i >= iter_max:
                break
        logger.info(f"Stopped at NMF iteration={i} loss={loss:.2e}")
        return loss_history

    def loss(self, pen_lambdas: list[float]):
        full_loss, losses = self.nmf.loss([], pen_lambdas)
        losses["penalties"] = losses["penalties_H"]
        del losses["penalties_W"]
        del losses["penalties_H"]
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
    def H(self) -> torch.Tensor:
        ret = self.nmf.H.detach().clone()
        ret[self.W_ignored_cols.flatten(), :] = 0
        return ret * self.V_norm_fac / self.W_norm_fac.T

    @H.setter
    def H(self, value: torch.Tensor):
        value[self.W_ignored_cols.flatten(), :] = 0
        self.nmf.H = value / self.V_norm_fac * self.W_norm_fac.T

    @property
    def W(self):
        return self.nmf.W * self.W_norm_fac

    @property
    def V(self):
        return self.nmf.V * self.V_norm_fac


def multistage(
    inputs,
    fs,
    hops: list[float],
    overlap: float,
    nmels: int,
    low_power_threshold: float,
    spec_power: float,
    divergence: Divergence,
    iter_max: float,
    dloss_min: float,
    carve_threshold: float,
    carve_blur_size: int,
    carve_min_duration: float,
    carve_max_slope: float,
    doplot: bool = False,
):
    learners: list[ActivationLearner] = []

    for hop_size in hops:
        win_size = hop_size * overlap

        logger.info(f"Starting round with {hop_size=}s, {win_size=}s")

        learner = ActivationLearner(
            inputs,
            fs=fs,
            n_mels=nmels,
            win_size=win_size,
            hop_size=hop_size,
            divergence=divergence,
            penalties=[],
            low_power_threshold=low_power_threshold,
            use_gpu=False,
            spec_power=spec_power,
            stft_win_func="barthann",
        )

        # carve and resize H from previous round
        if len(learners) > 0:
            new_H = carve.H_interpass_enhance(
                learners[-1].H,
                learner.H.shape,
                learners[-1].split_idx,
                carve_threshold,
                carve_blur_size,
                diag_size=max(2, int(carve_min_duration / hop_size)),
                max_slope=carve_max_slope,
                n_filters=15,
            )
            if doplot:
                plt.figure("H after resizing and carving")
                im=plot.plot_H(new_H.cpu().detach().numpy())
                plt.colorbar(im)
                plt.show()

            learner.H = new_H

        loss_history = learner.fit(iter_max, dloss_min=dloss_min)
        if doplot:
            plot.plot_nmf(learner)
            plt.show()
            plt.figure()
            plot.plot_loss_history(loss_history)
            plt.show()
        learners.append(learner)
    return learners[-1], loss_history
