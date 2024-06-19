import itertools
import logging
import os
import textwrap
import traceback
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import librosa
import numpy as np
import optuna
from optuna.artifacts import upload_artifact
import optuna_dashboard
from optuna_dashboard.artifact import get_artifact_path

import activation_learner
import modular_nmf
import param_estimator
import plot
from unmixdb import UnmixDB
from multiprocessing import Lock

UNMIXDB_PATH = Path("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")
unmixdb = UnmixDB(UNMIXDB_PATH)
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.CENTER_OF_MASS
LOG_NMF_EVERY = 100
DLOSS_MIN = 1e-3
ITER_MAX = 3000
TRYPRUNE_EVERY = 500
LAMBDA_MAX = 1e3
FS = 22050
MIX_NAME = "set275mix3-stretch-none-28.mp3"

mix = unmixdb.mixes[MIX_NAME]
input_paths = [unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks] + [
    mix.audio_path
]
pprint(input_paths)

# load audios
inputs = [librosa.load(path, sr=FS)[0] for path in input_paths]
base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)
lock = Lock()


def estimate(trial: optuna.trial.Trial, learner, mix, hop_size, step):
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

    # plots
    with lock:
        plot.plot_nmf(learner).savefig("temp.png")
        nmf_path = get_artifact_path(
            trial, upload_artifact(trial, "temp.png", artifact_store)
        )

        fig = plt.figure()
        plot.plot_gain(tau, est_gain, real_gain)
        fig.savefig("temp.png")
        gain_path = get_artifact_path(
            trial, upload_artifact(trial, "temp.png", artifact_store)
        )

        fig = plt.figure()
        plot.plot_warp(tau, est_warp, real_warp)
        fig.savefig("temp.png")
        warp_path = get_artifact_path(
            trial, upload_artifact(trial, "temp.png", artifact_store)
        )

    note = textwrap.dedent(f"""
    ![nmf]({nmf_path})
    
    ![gain]({gain_path})
    
    ![warp]({warp_path})
    
    """)
    optuna_dashboard.save_note(trial, note)

    plt.close("all")
    return err_gain, err_warp


def objective(trial: optuna.trial.Trial):
    hop_size = 1

    overlap = trial.suggest_float("overlap", 1, 8)
    win_size = hop_size * overlap
    nmels = 256
    low_power_factor = trial.suggest_float("low_power_factor", 1e-6, 1e2, log=True)
    divergence = modular_nmf.BetaDivergence(0)
    penalties = [
        (modular_nmf.SmoothDiago(), trial.suggest_float("smoothdiago", 0, LAMBDA_MAX)),
        (modular_nmf.L1(), trial.suggest_float("l1", 0, LAMBDA_MAX)),
        (modular_nmf.L2(), trial.suggest_float("l2", 0, LAMBDA_MAX)),
        (modular_nmf.SmoothGain(), trial.suggest_float("smoothgain", 0, LAMBDA_MAX)),
    ]

    try:
        logger = logging.getLogger(MIX_NAME)
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)
        logger.info(f"Starting round with {hop_size=}s, {win_size=}s")

        learner = activation_learner.ActivationLearner(
            inputs,
            fs=FS,
            n_mels=nmels,
            win_size=win_size,
            hop_size=hop_size,
            penalties=penalties,
            divergence=divergence,
            postprocessors=[],
            low_power_factor=low_power_factor,
        )

        # iterate
        logger.info(
            f"Running NMF on V:{learner._V.shape}, W:{learner._W.shape}, H:{learner._H.shape}"
        )
        last_loss = np.inf
        loss_history = []
        for i in itertools.count():
            loss, loss_components = learner.iterate(0)
            dloss = abs(last_loss - loss)
            last_loss = loss
            loss_history.append(loss_components)

            if i % LOG_NMF_EVERY == 0:
                logger.info(f"NMF iteration={i} loss={loss:.2e} dloss={dloss:.2e}")
            # if i % TRYPRUNE_EVERY == 0:
            # estimate(trial, learner, mix, hop_size, i)
            if dloss < DLOSS_MIN or i >= ITER_MAX:
                logger.info(f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}")
                break

        return estimate(trial, learner, mix, hop_size, i)
    except optuna.TrialPruned:
        raise
    except Exception:
        traceback.print_exc()
        return None


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="big_search_gain2",
    pruner=optuna.pruners.HyperbandPruner(),
    # sampler=optuna.samplers.RandomSampler(),
    load_if_exists=True,
    # err_gain, err_warp, time_nmf
    directions=["minimize", "minimize"],
)  # Create a new study.
study.optimize(
    objective,
    n_trials=1000,
    n_jobs=5,
)  # Invoke optimization of the objective function.
