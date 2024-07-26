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

import activation_learner
import param_estimator
import plot
import pytorch_nmf
from mixes.unmixdb import UnmixDB
from mixes.synthetic import SyntheticDB
import carve

# db = UnmixDB("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo/")
# db = UnmixDB("/data2/anasynth_nonbp/andre/unmixdb-zenodo")
db = SyntheticDB()
MIX_NAME = "linear-mix"
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.ARGMAX
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
    hops = [1, 0.5, 0.1]
    overlap = 4
    nmels = 256
    low_power_threshold = 0.1
    spec_power = 2
    learners: list[activation_learner.ActivationLearner] = []
    logger = logging.getLogger(MIX_NAME)
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    try:
        for hop_size in hops:
            win_size = hop_size * overlap
            divergence = pytorch_nmf.BetaDivergence(0)
            penalties = [
                (pytorch_nmf.L1(), trial.suggest_float("l1", 1e-4, 1e4, log=True)),
                (
                    pytorch_nmf.SmoothDiago(),
                    trial.suggest_float("diago", 1e-4, 1e4, log=True),
                ),
                (
                    pytorch_nmf.Lineness(),
                    trial.suggest_float("lineness", 1e-4, 1e4, log=True),
                ),
            ]

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
                use_gpu=USE_GPU,
                spec_power=spec_power,
                stft_win_func="blackman",
            )

            # carve and resize H from previous round
            if len(learners) > 0:
                new_H = carve.H_interpass_enhance(
                    learners[-1].H, learner.H.shape, 1e-3, 3, "bilinear"
                )
                plt.figure("H after resizing and carving")
                plot.plot_H(new_H.cpu().detach().numpy())
                plt.show()

                learner.H = new_H

            loss_history = learner.fit(2000)

            learners.append(learner)
        #############
        # estimations

        # get ground truth
        tau = np.arange(0, learner.V.shape[1]) * hop_size
        real_gain = mix.gain(tau)
        real_warp = mix.warp(tau)

        # estimate gain
        est_gain = GAIN_ESTOR(learner, spec_power)
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
