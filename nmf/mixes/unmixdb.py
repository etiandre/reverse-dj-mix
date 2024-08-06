import re
from pathlib import Path
from dataclasses import dataclass
import csv
from functools import cached_property
import librosa
import numpy as np
from typing import Union
from .classes import FromFileMix, FromFileRefTrack, Mix, RefTrack, Dataset


def read_meta(labels_path: Path):
    meta = [{}, {}, {}]
    with open(labels_path) as f:
        for row in csv.reader(f, delimiter="\t"):
            t0 = float(row[0])
            t1 = float(row[1])
            track_id = int(row[2]) - 1
            command = row[3]
            arg = row[4]
            if command == "start":
                meta[track_id]["name"] = arg
                meta[track_id]["start"] = t0
            elif command == "stop":
                meta[track_id]["stop"] = t0
            elif command == "speed":
                meta[track_id]["speed"] = float(arg)
            elif command == "bpm":
                meta[track_id]["bpm"] = float(arg)
            elif command == "fadein":
                meta[track_id]["fadein"] = (t0, t1)
            elif command == "fadeout":
                meta[track_id]["fadeout"] = (t0, t1)
            elif command == "cutpoint":
                meta[track_id]["cutpoint"] = t0
                pass
            else:
                raise RuntimeError(command)
    return meta


class UnmixDBTrack(FromFileRefTrack):
    def __init__(
        self, audio_path: Path, beat_path: Path, mfcc_path: Path, txt_path: Path
    ):
        super().__init__(audio_path)
        self.beat_path = beat_path
        self.mfcc_path = mfcc_path
        self.txt_path = txt_path

    @cached_property
    def meta(self):
        with open(self.txt_path) as f:
            data = f.readlines()[1].split("\t")
        return {
            "bpm": float(data[1]),
            "cueinstart": float(data[2]),
            "cueinend": float(data[3]),
            "cutpoint": float(data[4]),
            "joinpoint": float(data[5]),
            "cueoutstart": float(data[6]),
            "cueoutend": float(data[7]),
            "duration": float(data[8]),
        }


class UnmixDBMix(FromFileMix):
    def __init__(
        self,
        audio_path: Path,
        timestretch: str,
        fx: str,
        meta: list[dict],
        tracks: list[UnmixDBTrack],
    ):
        super().__init__(audio_path)
        self.timestretch = timestretch
        self.fx = fx
        self.meta = meta
        self._tracks = tracks

    @property
    def tracks(self):
        return self._tracks

    @property
    def audio_path(self):
        return self._audio_path

    @property
    def duration(self) -> float:
        return self.meta[-1]["stop"]

    def gain(self, times: np.ndarray):
        ret = []
        for track in self.meta:
            gain = np.zeros_like(times, dtype=float)
            for i, t in enumerate(times):
                if t < track["fadein"][0]:
                    g = 0
                elif t < track["fadein"][1]:
                    g = 1 + (t - track["fadein"][1]) / (
                        track["fadein"][1] - track["fadein"][0]
                    )

                elif t < track["fadeout"][0]:
                    g = 1
                elif t < track["fadeout"][1]:
                    g = -(t - track["fadeout"][1]) / (
                        track["fadeout"][1] - track["fadeout"][0]
                    )
                else:
                    g = 0
                # divide gain by number of tracks. Mixes have been generated with sox -m, cf. sox(1):
                # Unlike the other methods, ‘mix' combining has the potential to cause clipping in the combiner if no balancing is performed.  In this case, if manual volume adjustments are not  given,  SoX  will
                # try  to  ensure  that clipping does not occur by automatically adjusting the volume (amplitude) of each input signal by a factor of ¹/n, where n is the number of input files.  If this results in
                # audio that is too quiet or otherwise unbalanced then the input file volumes can be set manually as described above. Using the norm effect on the mix is another alternative.
                gain[i] = g / 3
            ret.append(gain)
        return np.stack(ret).T

    def warp(self, times: np.ndarray):
        ret = []
        for track in self.meta:
            position = np.zeros_like(times, dtype=float)
            for i, t in enumerate(times):
                if t < track["start"]:
                    tau = np.nan
                elif t < track["stop"]:
                    tau = track["speed"] * (t - track["start"])
                else:
                    tau = np.nan
                position[i] = tau
            ret.append(position)
        return np.stack(ret).T


class UnmixDB(Dataset):
    def __init__(self, base_path: Union[Path, str], only_good_mixes=True):
        self._mixes = []
        base_path = Path(base_path)

        assert base_path.exists()

        if only_good_mixes:
            with open(Path(__file__).parent / "unmixdb-goodmixes.txt") as f:
                good_mixes = [i.strip() for i in f.readlines()]

        for subset in base_path.glob("*"):
            reftracks = {}
            for audio_path in subset.glob("refsongs/*.mp3"):
                assert audio_path.name not in reftracks
                reftracks[audio_path.name] = UnmixDBTrack(
                    audio_path=audio_path,
                    mfcc_path=audio_path.with_suffix(".mp3-mfcc.mat"),
                    txt_path=audio_path.with_suffix(".txt"),
                    beat_path=audio_path.with_suffix("").with_suffix(".beat.xml"),
                )

            for audio_path in subset.glob("mixes/*.mp3"):
                if only_good_mixes and audio_path.name not in good_mixes:
                    print(f"skip bad mix {audio_path.name}")
                    continue
                if m := re.match(r"\w+-(\w+)-(\w+)-(\d+)", audio_path.stem):
                    timestretch = m.group(1)
                    fx = m.group(2)
                    labels_path = audio_path.with_suffix(".labels.txt")
                    meta = read_meta(labels_path)
                    tracks = [reftracks[i["name"]] for i in meta]
                    # id = int(m.group(3))
                    self._mixes.append(
                        UnmixDBMix(
                            audio_path=audio_path,
                            timestretch=timestretch,
                            fx=fx,
                            meta=meta,
                            tracks=tracks,
                        )
                    )
                else:
                    raise RuntimeError

    @property
    def mixes(self) -> list[UnmixDBMix]:
        return self._mixes

    @cached_property
    def timestretches(self):
        return set(i.timestretch for i in self.mixes)

    @cached_property
    def fxes(self):
        return set(i.fx for i in self.mixes)
