import re
from pathlib import Path
from dataclasses import dataclass
import csv
from functools import cached_property
import librosa

@dataclass
class Track:
    name: str

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
        tracks=[{}, {}, {}]
        with open(self.labels_path) as f:
            for row in csv.reader(f, delimiter='\t'):
                t0 = float(row[0])
                t1 = float(row[1])
                track_id=int(row[2])
                command = row[3]
                arg = row[4]
                if command == "start":
                    tracks[track_id-1]['name'] = arg
                    tracks[track_id-1]['start'] = t0
                elif command == "stop":
                    tracks[track_id-1]['stop'] = t0
                elif command == "speed":
                    tracks[track_id-1]['speed'] = float(arg)
                elif command == "bpm":
                    pass
                elif command=="fadein":
                    pass
                elif command=="fadeout":
                    pass
                elif command=="cutpoint":
                    tracks[track_id-1]['cutpoint'] = t0
                    pass
                else:
                    raise RuntimeError(command)
        return tracks
@dataclass
class RefSong:
    audio_path: Path
    beat_path: Path
    mfcc_path: Path
    txt_path: Path
    
    def audio(self, **kwargs):
        return librosa.load(self.audio_path, **kwargs)[0]


class UnmixDB:
    def __init__(self, base_path: Path|str):
        base_path=Path(base_path)
        self.mixes = {}
        self.refsongs = {}
        for subset in base_path.glob('*'):
            for audio_path in subset.glob('mixes/*.mp3'):
                if m := re.match(r"\w+-(\w+)-(\w+)-(\d+)", audio_path.stem):
                    timestretch = m.group(1)
                    fx = m.group(2)
                    # id = int(m.group(3))
                    self.mixes[audio_path.name] = Mix(audio_path=audio_path, labels_path=audio_path.with_suffix(".labels.txt"), mfcc_path=audio_path.with_suffix(".mp3-mfcc.mat"), timestretch=timestretch, fx=fx)
                else:
                    raise RuntimeError

            for audio_path in subset.glob('refsongs/*.mp3'):
                assert audio_path.name not in self.refsongs.keys()
                self.refsongs[audio_path.name] = RefSong(
                    audio_path=audio_path,
                    mfcc_path=audio_path.with_suffix(".mp3-mfcc.mat"),
                    txt_path=audio_path.with_suffix(".txt"),
                    beat_path=audio_path.with_suffix("").with_suffix(".beat.xml")
                )
    @cached_property
    def timestretches(self):
        return set(i.timestretch for i in self.mixes.values())
    @cached_property
    def fxes(self):
        return set(i.fx for i in self.mixes.values())