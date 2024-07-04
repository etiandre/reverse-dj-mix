import itertools
import logging
import os
import pickle
import textwrap
import traceback
from pathlib import Path
from pprint import pprint

import librosa
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna_dashboard
from optuna.artifacts import upload_artifact
from optuna_dashboard.artifact import get_artifact_path
from tqdm import tqdm


USE_GPU = False

if USE_GPU:
    import manage_gpus as gpl

    gpl.get_gpu_lock()
from multiprocessing import Lock

import torch.profiler

import activation_learner
import param_estimator
import plot
import pytorch_nmf
from mixes.unmixdb import UnmixDB
from mixes.synthetic import SyntheticDB

# db = UnmixDB("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo/")
# db = UnmixDB("/data2/anasynth_nonbp/andre/unmixdb-zenodo")
db = SyntheticDB()
MIX_NAME = "linear-mix"
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.CENTER_OF_MASS
LOG_NMF_EVERY = 100
DLOSS_MIN = -np.inf
ITER_MAX = 1000
FS = 22050


mix = db.get_mix(MIX_NAME)

# load audios
inputs = mix.as_activation_learner_input()
base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)
lock = Lock()


def objective(trial: optuna.trial.Trial):
    hop_size = 0.1

    overlap = trial.suggest_float("overlap", 1, 8)
    win_size = hop_size * overlap
    nmels = 512
    low_power_threshold = 0.1
    divergence = pytorch_nmf.BetaDivergence(1)
    penalties = [
        (pytorch_nmf.L1(), trial.suggest_float("l1", 1e-4, 1e4, log=True)),
        (
            pytorch_nmf.SmoothDiago(),
            trial.suggest_float("smoothdiago", 1e-4, 1e4, log=True),
        ),
        (
            pytorch_nmf.SmoothOverCol(),
            trial.suggest_float("smoothovercol", 1e-4, 1e4, log=True),
        ),
        (
            pytorch_nmf.SmoothOverRow(),
            trial.suggest_float("smoothoverrow", 1e-4, 1e4, log=True),
        ),
        (pytorch_nmf.Lineness(), trial.suggest_float("lineness", 1e-4, 1e6, log=True)),
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
            low_power_threshold=low_power_threshold,
            spec_power=1,
            use_gpu=USE_GPU,
        )
        #################
        # iterate
        logger.info(
            f"Running NMF on V:{learner.nmf.V.shape}, W:{learner.nmf.W.shape}, H:{learner.nmf.H.shape}"
        )
        loss_history = learner.fit(ITER_MAX)

        #############
        # estimations

        # get ground truth
        tau = np.arange(0, learner.V.shape[1]) * hop_size
        real_gain = mix.gain(tau)
        real_warp = mix.warp(tau)

        # estimate gain
        est_gain = GAIN_ESTOR(learner)
        err_gain = param_estimator.error(est_gain, real_gain)

        # estimate warp
        est_warp = WARP_ESTOR(learner, hop_size)
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

        with lock:
            with open("temp.pickle", "wb") as f:
                pickle.dump(learner, f)
            model_path = get_artifact_path(
                trial, upload_artifact(trial, "temp.pickle", artifact_store)
            )
        note = textwrap.dedent(f"""
        [model.pickle]({model_path})
        
        ![loss]({loss_path})
        
        ![nmf]({nmf_path})
        
        ![gain]({gain_path})
        
        ![warp]({warp_path})
        """)
        optuna_dashboard.save_note(trial, note)

        plt.close("all")
        return loss_history[-1]["divergence"], err_gain, err_warp
    except optuna.TrialPruned:
        raise
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return None


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    # study_name="pouet2",
    # pruner=optuna.pruners.HyperbandPruner(),
    sampler=optuna.samplers.NSGAIIISampler(),
    load_if_exists=True,
    # err_gain, err_warp, time_nmf
    directions=["minimize", "minimize", "minimize"],
)  # Create a new study.
study.optimize(
    objective,
    n_trials=2000,
    n_jobs=18,
)  # Invoke optimization of the objective function.
