import optuna
import logging
from pathlib import Path
import numpy as np
import pydub
import matplotlib.pyplot as plt
import librosa
from pprint import pprint
import scipy.ndimage
import scipy.signal
import logging
import itertools
import sparse
import datetime
import os
import pickle
import time
import joblib

from unmixdb import UnmixDB
from abcdj import ABCDJ
import activation_learner, carve, plot, param_estimator, util, modular_nmf

from common import dense_to_sparse, sparse_to_dense


UNMIXDB_PATH = Path("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")
unmixdb = UnmixDB(UNMIXDB_PATH)
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.CENTER_OF_MASS
LOG_NMF_EVERY = 500
ITER_MAX = 2000
OVERLAP_FACTOR = 1
FS = 22050
NMELS = 256


def run_learner(mix_name, mix, hop_size, divergence, penalties):
    logger = logging.getLogger(mix_name)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Starting work on {mix_name}")
    input_paths = [
        unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks
    ] + [mix.audio_path]
    pprint(input_paths)

    # load audios
    inputs = [librosa.load(path, sr=FS)[0] for path in input_paths]

    win_size = OVERLAP_FACTOR * hop_size
    logger.info(f"Starting round with {hop_size=}s, {win_size=}s")

    learner = activation_learner.ActivationLearner(
        inputs,
        fs=FS,
        n_mels=NMELS,
        win_size=win_size,
        hop_size=hop_size,
        penalties=penalties,
        divergence=divergence,
        postprocessors=[],
    )

    # iterate
    logger.info("Running NMF")
    last_loss = np.inf
    loss_history = []
    for i in itertools.count():
        loss, loss_components = learner.iterate(0)
        dloss = abs(last_loss - loss)
        last_loss = loss
        loss_history.append(loss_components)

        if i % LOG_NMF_EVERY == 0:
            logger.info(f"NMF iteration={i} loss={loss:.2e} dloss={dloss:.2e}")
        if i > ITER_MAX:
            logger.info(f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}")
            break

    # get ground truth
    tau = np.arange(0, learner.V.shape[1]) * hop_size
    real_gain = mix.get_track_gain(tau)
    # real_warp = mix.get_track_warp(tau)

    # estimate gain
    est_gain = GAIN_ESTOR.value(learner)
    err_gain = param_estimator.error(est_gain, real_gain)

    # estimate warp
    # est_warp = WARP_ESTOR.value(learner)
    # err_warp = param_estimator.error(est_warp, real_warp)

    return err_gain


def run_full_unmixdb(hop_size, beta, l1_fac, smoothgain_fac):
    mixes = dict(
        filter(
            lambda i: i[1].timestretch == "none" and i[1].fx == "none",
            unmixdb.mixes.items(),
        )
    )
    logging.info(f"Will process {len(mixes)} mixes")
    joblib.Parallel(15, verbose=10)(
        joblib.delayed(run_learner)(
            mix_name, mix, hop_size, beta, l1_fac, smoothgain_fac
        )
        for mix_name, mix in mixes.items()
    )


def objective(trial: optuna.trial.Trial):
    hop_size = 2

    beta = 0
    l1_fac = trial.suggest_float("l1_fac", 0, 1e30)
    smoothgain_fac = trial.suggest_float("smoothgain_fac", 0, 1e30)
    smoothdiago_fac = trial.suggest_float("smoothdiago_fac", 0, 1e30)

    divergence = modular_nmf.BetaDivergence(beta)
    penalties = [
        (modular_nmf.L1(), l1_fac),
        (modular_nmf.SmoothGain(), smoothgain_fac),
        (modular_nmf.SmoothDiago(), smoothdiago_fac),
    ]
    mix_name = "set275mix3-none-none-28.mp3"
    mix = unmixdb.mixes[mix_name]
    try:
        err_gain = run_learner(mix_name, mix, hop_size, divergence, penalties)
        return err_gain
    except Exception as e:
        return None


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="pouet1",
    load_if_exists=True,
)  # Create a new study.
study.optimize(
    objective, n_trials=100
)  # Invoke optimization of the objective function.
