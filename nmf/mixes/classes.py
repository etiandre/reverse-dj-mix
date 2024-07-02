import abc
from functools import cached_property
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import librosa

FS = 22050


class RefTrack(abc.ABC):
    @property
    @abc.abstractmethod
    def audio(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def duration(self):
        return len(self.audio) / FS


class FromFileRefTrack(RefTrack):
    def __init__(self, audio_path: Union[Path, str]) -> None:
        self._audio_path = Path(audio_path)
        assert self._audio_path.exists()

    @property
    def audio_path(self) -> Path:
        return self._audio_path

    @cached_property
    def audio(self) -> np.ndarray:
        return librosa.load(self.audio_path, sr=FS)[0]

    @property
    def name(self) -> str:
        return self.audio_path.name


class Mix(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def audio(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gain(self, times: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def warp(self, times: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def tracks(self) -> Sequence[RefTrack]:
        pass

    @property
    def duration(self):
        return len(self.audio) / FS


class FromFileMix(Mix):
    def __init__(self, audio_path: Union[Path, str]) -> None:
        self._audio_path = Path(audio_path)
        assert self._audio_path.exists()

    @property
    def audio_path(self) -> Path:
        return self._audio_path

    @cached_property
    def audio(self) -> np.ndarray:
        return librosa.load(self.audio_path, sr=FS)[0]

    @property
    def name(self) -> str:
        return self.audio_path.name


class Dataset(abc.ABC):
    @property
    @abc.abstractmethod
    def mixes(self) -> Sequence[Mix]:
        pass

    def get_mix(self, name: str):
        for i in self.mixes:
            if i.name == name:
                return i
        raise ValueError(f"Cannot find mix '{name}'")
