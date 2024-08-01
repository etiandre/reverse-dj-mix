import datetime
import itertools
import json
import logging
import os
import pickle
from random import shuffle
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns

import activation_learner
import param_estimator
import plot
from mixes.unmixdb import UnmixDB, UnmixDBMix
from pytorch_nmf import BetaDivergence

# configuration
# =============

# hyperparams
FS = 22050
HOP_SIZES = [4, 2, 1, 0.5]
OVERLAP = 8
NMELS = 128
SPEC_POWER = 2
DIVERGENCE = BetaDivergence(0)
GAIN_ESTOR = param_estimator.GainEstimator.SUM
WARP_ESTOR = param_estimator.WarpEstimator.ARGMAX
LOW_POWER_THRESHOLD = 1e-2
CARVE_THRESHOLD = 1e-6
CARVE_BLUR_SIZE = 3
CARVE_MIN_DURATION = 10
CARVE_MAX_SLOPE = 1.5
# stop conditions
DLOSS_MIN = 1e-10
ITER_MAX = 5000
# paths
RESULTS_DIR = Path("/data5/anasynth_nonbp/andre/reverse-dj-mix/results")
UNMIXDB_PATH = Path("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")
#############################

# plt.style.use("dark_background")
date = datetime.datetime.now().isoformat()
os.makedirs(RESULTS_DIR / f"{date}")

logFormatter = logging.Formatter(
    "%(asctime)s [%(name)-23.23s] [%(levelname)-5.5s]  %(message)s"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
fileHandler = logging.FileHandler(RESULTS_DIR / f"{date}/output.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

unmixdb = UnmixDB(UNMIXDB_PATH)


def worker(mix: UnmixDBMix):
    results = {}
    os.makedirs(RESULTS_DIR / f"{date}/{mix.name}")
    sns.set_theme("paper")

    try:
        logger.info(f"Starting work on {mix.name}")
        results["mix.name"] = mix.name

        inputs = [track.audio for track in mix.tracks] + [mix.audio]

        tick = time.time()

        learner, loss_history = activation_learner.multistage(
            inputs,
            FS,
            hops=HOP_SIZES,
            overlap=OVERLAP,
            nmels=NMELS,
            low_power_threshold=LOW_POWER_THRESHOLD,
            spec_power=SPEC_POWER,
            divergence=DIVERGENCE,
            iter_max=ITER_MAX,
            dloss_min=DLOSS_MIN,
            carve_threshold=CARVE_THRESHOLD,
            carve_blur_size=CARVE_BLUR_SIZE,
            carve_min_duration=CARVE_MIN_DURATION,
            carve_max_slope=CARVE_MAX_SLOPE,
        )

        # get ground truth
        tau = np.arange(0, learner.V.shape[1]) * HOP_SIZES[-1]
        real_gain = mix.gain(tau)
        real_warp = mix.warp(tau)

        # estimate gain
        logger.info(f"Estimating gain with method {GAIN_ESTOR}")
        est_gain = GAIN_ESTOR(learner.H, learner.split_idx, SPEC_POWER)

        # estimate warp
        logger.info(f"Estimating warp with method {WARP_ESTOR}")
        est_warp = WARP_ESTOR(learner.H, learner.split_idx, HOP_SIZES[-1])

        tock = time.time()

        # save figures
        fig = plt.figure()
        plot.plot_gain(tau, est_gain, real_gain)
        fig.savefig(
            RESULTS_DIR / f"{date}/{mix.name}/{GAIN_ESTOR.__class__.__name__}.png"
        )

        fig = plt.figure()
        plot.plot_warp(tau, est_warp, real_warp)
        fig.savefig(
            RESULTS_DIR / f"{date}/{mix.name}/{WARP_ESTOR.__class__.__name__}.png"
        )

        plot.plot_nmf(learner).savefig(RESULTS_DIR / f"{date}/{mix.name}/nmf.png")

        fig = plt.figure()
        plot.plot_loss_history(loss_history)
        fig.savefig(RESULTS_DIR / f"{date}/{mix.name}/loss.png")

        # save results
        results["gain_real"] = real_gain
        results["warp_real"] = real_warp
        results["gain_est"] = est_gain
        results["warp_est"] = est_warp
        results["gain_err"] = param_estimator.error(est_gain, real_gain)
        results["warp_err"] = param_estimator.error(est_warp, real_warp)
        results["H"] = learner.H.detach().numpy()
        results["tau"] = tau
        results["time"] = tock - tick

    except Exception as e:
        if e is KeyboardInterrupt:
            exit(1)
        logger.exception(f"Error when processing {mix.name}, aborting")
    finally:
        with open(RESULTS_DIR / f"{date}/{mix.name}/results.pickle", "wb") as f:
            pickle.dump(results, f)
        scipy.io.savemat(RESULTS_DIR / f"{date}/{mix.name}/results.mat", results)
        plt.close("all")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workers", type=int, required=True)
    args = parser.parse_args()

    logging.info(f"Will process {len(unmixdb.mixes)} mixes")

    with open(RESULTS_DIR / f"{date}/meta.json", "w") as f:
        json.dump(
            {
                "FS": FS,
                "HOP_SIZES": HOP_SIZES,
                "OVERLAP": OVERLAP,
                "NMELS": NMELS,
                "SPEC_POWER": SPEC_POWER,
                "DIVERGENCE": str(DIVERGENCE),
                "GAIN_ESTOR": GAIN_ESTOR.__class__.__name__,
                "WARP_ESTOR": WARP_ESTOR.__class__.__name__,
                "LOW_POWER_THRESHOLD": LOW_POWER_THRESHOLD,
                "CARVE_THRESHOLD": CARVE_THRESHOLD,
                "CARVE_BLUR_SIZE": CARVE_BLUR_SIZE,
                "CARVE_MIN_DURATION": CARVE_MIN_DURATION,
                "CARVE_MAX_SLOPE": CARVE_MAX_SLOPE,
                "DLOSS_MIN": DLOSS_MIN,
                "ITER_MAX": ITER_MAX,
                "RESULTS_DIR": str(RESULTS_DIR.resolve()),
                "UNMIXDB_PATH": str(UNMIXDB_PATH.resolve()),
            },
            f,
        )
    if args.workers == 1:
        for mix in unmixdb.mixes:
            worker(mix)
    else:
        joblib.Parallel(args.workers, verbose=10)(
            joblib.delayed(worker)(mix) for mix in unmixdb.mixes
        )
