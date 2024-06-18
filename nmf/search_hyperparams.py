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
DLOSS_MIN = 1e-7

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
    hop_size = trial.suggest_float("hop_size", 0.01, 5)

    overlap = trial.suggest_float("overlap", 1, 8)
    win_size = hop_size * overlap
    nmels = trial.suggest_int("nmels", 64, 1024)
    low_power_factor = trial.suggest_float("low_power_factor", 1e-6, 1e2, log=True)
    fs = 22050
    divergence = modular_nmf.BetaDivergence(0)
    penalties = [
        (modular_nmf.SmoothDiago(), trial.suggest_float("smoothdiago", 0, 1e8)),
        (modular_nmf.L1(), trial.suggest_float("l1", 0, 1e8)),
        (modular_nmf.L2(), trial.suggest_float("l2", 0, 1e8)),
        (modular_nmf.SmoothGain(), trial.suggest_float("smoothgain", 0, 1e8)),
    ]
    mix_name = "set275mix3-stretch-none-28.mp3"
    
    
    mix = unmixdb.mixes[mix_name]
    try:
        logger = logging.getLogger(mix_name)
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)
        logger.info(f"Starting work on {mix_name}")
        input_paths = [
            unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks
        ] + [mix.audio_path]
        pprint(input_paths)

        # load audios
        inputs = [librosa.load(path, sr=fs)[0] for path in input_paths]

        logger.info(f"Starting round with {hop_size=}s, {win_size=}s")

        tick_start = time.time()
        learner = activation_learner.ActivationLearner(
            inputs,
            fs=fs,
            n_mels=nmels,
            win_size=win_size,
            hop_size=hop_size,
            penalties=penalties,
            divergence=divergence,
            postprocessors=[],
            low_power_factor=low_power_factor,
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
            if dloss < DLOSS_MIN:
                logger.info(f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}")
                break
        tick_stop = time.time()
        # get ground truth
        tau = np.arange(0, learner.V.shape[1]) * hop_size
        real_gain = mix.get_track_gain(tau)
        real_warp = mix.get_track_warp(tau)

        # estimate gain
        est_gain = GAIN_ESTOR.value(learner)
        err_gain = param_estimator.error(est_gain, real_gain)

        # estimate warp
        est_warp = WARP_ESTOR.value(learner, hop_size)
        err_warp = param_estimator.error(est_warp, real_warp)

        time_nmf = tick_start - tick_stop

        return err_gain, err_warp, time_nmf

    except Exception as e:
        return None


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="big_search",
    # sampler=optuna.samplers.RandomSampler(),
    load_if_exists=True,
    # err_gain, err_warp, time_nmf
    directions=["minimize", "minimize", "minimize"],
)  # Create a new study.
study.optimize(
    objective,
    n_trials=1000,
    n_jobs=10,
)  # Invoke optimization of the objective function.
