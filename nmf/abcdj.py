from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


@dataclass
class ABCDJResult:
    mat_path: Path
    rms_mat_path: Path

    @cached_property
    def data(self) -> dict:
        data = scipy.io.loadmat(self.mat_path, simplify_cells=True)
        data.update(scipy.io.loadmat(self.rms_mat_path, simplify_cells=True))
        return data

    @property
    def gain(self):
        # from df_fade_estim.m (abc-dj unmixing):
        #   the track is mixed such as mix = mix_factor * fade_curve * track
        gain = []
        for i in range(3):
            fadecurve = self.data["fadecurve"][i]
            mixfactor = self.data["mixfactor"][i]
            gain.append(fadecurve * mixfactor)

    @property
    def tau(self):
        return np.arange(len(self.data["fadecurve"][0])) * self.data["hop"]


def ABCDJ(base_path: Path | str) -> dict[str, ABCDJResult]:
    base_path = Path(base_path)
    ret = {}
    for rms_mat_path in base_path.glob("*-rms.mat"):
        name = rms_mat_path.name.replace("-rms.mat", "")
        mat_path = Path(str(rms_mat_path).replace("-rms.mat", ".mat"))
        ret[name] = ABCDJResult(mat_path, rms_mat_path)
    return ret
