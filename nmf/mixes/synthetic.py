from functools import cached_property
from pathlib import Path
from typing import Union
from .classes import FromFileMix, FromFileRefTrack, Mix, RefTrack, Dataset, FS
import numpy as np
import pyrubberband


class CrossfadeMix(Mix):
    def __init__(
        self,
        name: str,
        A: RefTrack,
        B: RefTrack,
        fade_start: float,
        fade_stop: float,
    ):
        self._tracks = [A, B]
        self.fade_start = fade_start
        self.fade_stop = fade_stop
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def tracks(self):
        return self._tracks

    @cached_property
    def audio(self):
        n_fade_start = int(self.fade_start * FS)
        n_fade_stop = int(self.fade_stop * FS)
        n_fade = n_fade_stop - n_fade_start
        curve = np.linspace(0, 1, n_fade)

        A = self.tracks[0]
        A_fade = A.audio[-n_fade:] * (1 - curve)
        B = self.tracks[1]
        B_fade = B.audio[:n_fade] * curve

        return np.concatenate([A.audio[:-n_fade], A_fade + B_fade, B.audio[n_fade:]])

    def gain(self, times: np.ndarray) -> np.ndarray:
        gain_B = np.zeros_like(times, dtype=float)
        for i, t in enumerate(times):
            if t <= self.fade_start:
                g = 0
            elif t <= self.fade_stop:
                g = 1 + (t - self.fade_stop) / (self.fade_stop - self.fade_start)
            else:
                g = 1
            gain_B[i] = g

        gain_A = np.zeros_like(times, dtype=float)
        for i, t in enumerate(times):
            if t <= self.fade_start:
                g = 1
            elif t <= self.fade_stop:
                g = 1 + (t - self.fade_start) / (self.fade_start - self.fade_stop)
            else:
                g = 0
            gain_A[i] = g

        return np.stack((gain_A, gain_B)).T

    def warp(self, times: np.ndarray) -> np.ndarray:
        warp_A = np.zeros_like(times, dtype=float) * np.nan
        for i, t in enumerate(times):
            if t <= self.fade_stop:
                warp_A[i] = t

        warp_B = np.zeros_like(times, dtype=float) * np.nan
        for i, t in enumerate(times):
            if t >= self.fade_start:
                warp_B[i] = t - self.fade_start

        return np.stack(arrays=(warp_A, warp_B)).T


class TimestretchMix(Mix):
    def __init__(
        self,
        name: str,
        track: RefTrack,
        timemap: list[tuple[float, float]],
        duration: float,
    ):
        self._name = name
        self._track = track
        timemap.append((track.duration, duration))
        self._timemap = timemap

    @property
    def name(self):
        return self._name

    @cached_property
    def audio(self) -> np.ndarray:
        timemap_samples = [(int(a * FS), int(b * FS)) for a, b in self._timemap]
        return pyrubberband.timemap_stretch(self._track.audio, FS, timemap_samples)

    def gain(self, times: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.ones_like(times)).T

    def warp(self, times: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.interp(times, *zip(*self._timemap))).T

    @property
    def tracks(self):
        return [self._track]


class SyntheticDB(Dataset):
    @property
    def mixes(self):
        DEADMAU5_A = FromFileRefTrack(Path(__file__).parent / "audio/linear-mix-1.wav")
        DEADMAU5_B = FromFileRefTrack(Path(__file__).parent / "audio/linear-mix-2.wav")
        NUTTAH = FromFileRefTrack(Path(__file__).parent / "audio/nuttah.wav")

        return [
            CrossfadeMix("linear-mix", DEADMAU5_A, DEADMAU5_B, 3.75, 7.5),
            CrossfadeMix("linear-mix-desync", DEADMAU5_A, DEADMAU5_B, 3.6, 7.5),
            CrossfadeMix("nuttah-deadmau5", NUTTAH, DEADMAU5_B, 2, 5),
            TimestretchMix("stretch", DEADMAU5_A, [(0, 0), (2,2), (3, 5)], 8),
        ]
