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
from tqdm import tqdm

import activation_learner
import pytorch_nmf
import param_estimator
import plot
from unmixdb import UnmixDB
from multiprocessing import Lock

UNMIXDB_PATH = Path("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")
unmixdb = UnmixDB(UNMIXDB_PATH)
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.CENTER_OF_MASS
LOG_NMF_EVERY = 100
DLOSS_MIN = -np.inf
ITER_MAX = 1000
TRYPRUNE_EVERY = 500
LAMBDA_MAX = 1e5
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


def objective(trial: optuna.trial.Trial):
    hop_size = 1

    overlap = trial.suggest_float("overlap", 1, 8)
    win_size = hop_size * overlap
    nmels = 64
    low_power_factor = trial.suggest_float("low_power_factor", 1e-4, 1e4, log=True)
    divergence = pytorch_nmf.ItakuraSaito()
    penalties = [
        (pytorch_nmf.L1(), trial.suggest_float("l1", 0, 1e4)),
        (pytorch_nmf.L2(), trial.suggest_float("l2", 0, 1e4)),
        (pytorch_nmf.SmoothOverCol(), trial.suggest_float("smoothcol", 0, 1e3)),
        (
            pytorch_nmf.SmoothOverRow(),
            trial.suggest_float("smoothrow", 0, 1e3),
        ),
        (pytorch_nmf.SmoothDiago(), trial.suggest_float("smoothdiago", 0, 1e3)),
        (
            pytorch_nmf.Lineness(),
            trial.suggest_float("lineness", 0, 1e6),
        ),
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
            low_power_factor=low_power_factor,
        )
        #################
        # iterate
        logger.info(
            f"Running NMF on V:{learner.nmf.V.shape}, W:{learner.nmf.W.shape}, H:{learner.nmf.H.shape}"
        )
        loss_history = []
        loss = np.inf
        for i in tqdm(itertools.count(), desc=f"Trial {trial.number}", total=ITER_MAX):
            learner.iterate()
            if i % 10 == 0:
                loss, loss_components = learner.loss()
                loss_history.append(loss_components)

                if i >= ITER_MAX:
                    logger.info(f"Stopped at NMF iteration={i} loss={loss}")
                    break
        #############
        # estimations

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

        ###################
        # plots
        with lock:
            plot.plot_nmf(learner).savefig("temp.png")
            nmf_path = get_artifact_path(
                trial, upload_artifact(trial, "temp.png", artifact_store)
            )

        with lock:
            fig = plt.figure()
            plot.plot_gain(tau, est_gain, real_gain)
            fig.savefig("temp.png")
            gain_path = get_artifact_path(
                trial, upload_artifact(trial, "temp.png", artifact_store)
            )

        with lock:
            fig = plt.figure()
            plot.plot_warp(tau, est_warp, real_warp)
            fig.savefig("temp.png")
            warp_path = get_artifact_path(
                trial, upload_artifact(trial, "temp.png", artifact_store)
            )

        with lock:
            fig = plt.figure()
            plot.plot_loss_history(loss_history)
            fig.savefig("temp.png")
            loss_path = get_artifact_path(
                trial, upload_artifact(trial, "temp.png", artifact_store)
            )

        note = textwrap.dedent(f"""
        ![loss]({loss_path})
        
        ![nmf]({nmf_path})
        
        ![gain]({gain_path})
        
        ![warp]({warp_path})
        
        """)
        optuna_dashboard.save_note(trial, note)

        plt.close("all")
        return loss, err_gain, err_warp
    except optuna.TrialPruned:
        raise
    except Exception:
        traceback.print_exc()
        return None


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    # study_name="pouet2",
    pruner=optuna.pruners.HyperbandPruner(),
    sampler=optuna.samplers.NSGAIIISampler(),
    load_if_exists=True,
    # err_gain, err_warp, time_nmf
    directions=["minimize", "minimize", "minimize"],
)  # Create a new study.
study.optimize(
    objective,
    n_trials=2000,
    n_jobs=20,
)  # Invoke optimization of the objective function.
