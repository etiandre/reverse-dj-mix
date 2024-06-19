import re
from pathlib import Path
from dataclasses import dataclass
import csv
from functools import cached_property
import librosa
import numpy as np


@dataclass
class Mix:
    audio_path: Path
    labels_path: Path
    mfcc_path: Path
    timestretch: str
    fx: str

    def audio(self, **kwargs):
        return librosa.load(self.audio_path, **kwargs)[0]

    @cached_property
    def tracks(self):
        tracks = [{}, {}, {}]
        with open(self.labels_path) as f:
            for row in csv.reader(f, delimiter="\t"):
                t0 = float(row[0])
                t1 = float(row[1])
                track_id = int(row[2]) - 1
                command = row[3]
                arg = row[4]
                if command == "start":
                    tracks[track_id]["name"] = arg
                    tracks[track_id]["start"] = t0
                elif command == "stop":
                    tracks[track_id]["stop"] = t0
                elif command == "speed":
                    tracks[track_id]["speed"] = float(arg)
                elif command == "bpm":
                    pass
                elif command == "fadein":
                    tracks[track_id]["fadein"] = (t0, t1)
                elif command == "fadeout":
                    tracks[track_id]["fadeout"] = (t0, t1)
                elif command == "cutpoint":
                    tracks[track_id]["cutpoint"] = t0
                    pass
                else:
                    raise RuntimeError(command)
        return tracks

    @cached_property
    def duration(self):
        return self.tracks[-1]["stop"]

    def get_track_gain(self, times: np.ndarray):
        ret = []
        for track in self.tracks:
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
                gain[i] = g / len(self.tracks)
            ret.append(gain)
        return np.stack(ret).T

    def get_track_warp(self, times: np.ndarray):
        ret = []
        for track in self.tracks:
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


@dataclass
class RefSong:
    audio_path: Path
    beat_path: Path
    mfcc_path: Path
    txt_path: Path

    def audio(self, **kwargs):
        return librosa.load(self.audio_path, **kwargs)[0]

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


class UnmixDB:
    def __init__(self, base_path: Path | str):
        base_path = Path(base_path)
        self.mixes: dict[str, Mix] = {}
        self.refsongs: dict[str, RefSong] = {}
        assert base_path.exists()
        for subset in base_path.glob("*"):
            for audio_path in subset.glob("mixes/*.mp3"):
                if m := re.match(r"\w+-(\w+)-(\w+)-(\d+)", audio_path.stem):
                    timestretch = m.group(1)
                    fx = m.group(2)
                    # id = int(m.group(3))
                    self.mixes[audio_path.name] = Mix(
                        audio_path=audio_path,
                        labels_path=audio_path.with_suffix(".labels.txt"),
                        mfcc_path=audio_path.with_suffix(".mp3-mfcc.mat"),
                        timestretch=timestretch,
                        fx=fx,
                    )
                else:
                    raise RuntimeError

            for audio_path in subset.glob("refsongs/*.mp3"):
                assert audio_path.name not in self.refsongs.keys()
                self.refsongs[audio_path.name] = RefSong(
                    audio_path=audio_path,
                    mfcc_path=audio_path.with_suffix(".mp3-mfcc.mat"),
                    txt_path=audio_path.with_suffix(".txt"),
                    beat_path=audio_path.with_suffix("").with_suffix(".beat.xml"),
                )

    @cached_property
    def timestretches(self):
        return set(i.timestretch for i in self.mixes.values())

    @cached_property
    def fxes(self):
        return set(i.fx for i in self.mixes.values())

    def get_refsongs_for(self, mix: Mix) -> list[RefSong]:
        return [self.refsongs[i["name"]] for i in mix.tracks]
