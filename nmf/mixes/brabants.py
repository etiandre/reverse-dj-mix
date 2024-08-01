from pathlib import Path
from typing import Union
from .classes import FromFileMix, FromFileRefTrack, Mix, RefTrack, Dataset
import numpy as np


class BrabantsTrack(FromFileRefTrack):
    def __init__(self, audio_path: Path | str) -> None:
        super().__init__(audio_path)


class BrabantsMix(FromFileMix):
    def __init__(self, audio_path: Path | str, tracks: list[BrabantsTrack]) -> None:
        super().__init__(audio_path)
        self._tracks = tracks

    @property
    def tracks(self):
        return self._tracks

    def gain(self, times: np.ndarray) -> np.ndarray:
        return np.zeros_like(times)

    def warp(self, times: np.ndarray) -> np.ndarray:
        return np.zeros_like(times)


class Brabants(Dataset):
    def __init__(self, base_path: Union[Path, str]):
        base_path = Path(base_path)
        assert base_path.exists()

        self._mixes = []
        for i in base_path.glob("*"):
            mix_path = i / f"n_{i.name}.wav"
            assert mix_path.exists()

            tracks = []
            for j in (i / "refsongs").glob("*.wav"):
                tracks.append(BrabantsTrack(j))

            self._mixes.append(BrabantsMix(mix_path, tracks))

    @property
    def mixes(self):
        return self._mixes
