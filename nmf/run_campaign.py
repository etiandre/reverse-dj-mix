import datetime
import itertools
import logging
import os
import pickle
import time
from pathlib import Path
from pprint import pprint

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import activation_learner
import carve
import modular_nmf
import param_estimator
import plot
from common import dense_to_sparse
from unmixdb import UnmixDB

# configuration
# =============

# hyperparams
FS = 22050
HOP_SIZES = [1.0]
OVERLAP_FACTOR = 1
CARVE_THRESHOLD_DB = -60
NMELS = 256
DIVERGENCE = modular_nmf.BetaDivergence(0)
PENALTIES = [
    (modular_nmf.SmoothDiago(), 10000),
    # (modular_nmf.L1(), 10),
    # (modular_nmf.SmoothGain(), 10),
    # (modular_nmf.VirtanenTemporalContinuity(), 1)
]
POSTPROCESSORS = [
    # (modular_nmf.PolyphonyLimit(1), 0.1)
]
PP_STRENGTH = 1
# stop conditions
DLOSS_MIN = -np.inf
LOSS_MIN = -np.inf
ITER_MAX = 3000
## other stuff
# logging
PLOT_NMF_EVERY = 600
LOG_NMF_EVERY = 100
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
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

unmixdb = UnmixDB(UNMIXDB_PATH)
# unmixdb = UnmixDB("/home/etiandre/stage/datasets/unmixdb-zenodo")


def worker(mix_name, mix):
    results = {}
    os.makedirs(RESULTS_DIR / f"{date}/{mix_name}")
    sns.set_theme("paper")

    # setup logging
    logger = logging.getLogger(mix_name)
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(RESULTS_DIR / f"{date}/{mix_name}/output.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    try:
        logger.info(f"Starting work on {mix_name}")
        results["mix_name"] = mix_name
        results["hyperparams"] = {
            "FS": FS,
            "HOP_SIZES": HOP_SIZES,
            "OVERLAP_FACTOR": OVERLAP_FACTOR,
            "CARVE_THRESHOLD_DB": CARVE_THRESHOLD_DB,
            "DLOSS_MIN": DLOSS_MIN,
            "LOSS_MIN": LOSS_MIN,
            "ITER_MAX": ITER_MAX,
            "NMELS": NMELS,
            "DIVERGENCE": DIVERGENCE,
            "PENALTIES": PENALTIES,
        }
        logger.info(results["hyperparams"])
        input_paths = [
            unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks
        ] + [mix.audio_path]
        for i in input_paths:
            logger.info(f"input path: {i}")

        tick_init = time.time()

        # load audios
        inputs = [librosa.load(path, sr=FS)[0] for path in input_paths]

        tick_load = time.time()

        # multi pass NMF
        previous_H = None
        previous_split_idx = None
        for hop_size in HOP_SIZES:
            win_size = OVERLAP_FACTOR * hop_size
            logger.info(f"Starting round with {hop_size=}s, {win_size=}s")

            learner = activation_learner.ActivationLearner(
                inputs,
                fs=FS,
                n_mels=NMELS,
                win_size=win_size,
                hop_size=hop_size,
                divergence=DIVERGENCE,
                penalties=PENALTIES,
                postprocessors=POSTPROCESSORS,
            )

            # carve and resize H from previous round
            if previous_H is not None:
                H_carved = carve.carve(
                    previous_H, previous_split_idx, CARVE_THRESHOLD_DB
                )
                H_carved_resized = carve.resize_cv_area(H_carved, learner.H.shape)
                plot.plot_carve_resize(H_carved, H_carved_resized).savefig(
                    RESULTS_DIR / f"{date}/{mix_name}/carve-{hop_size}.png"
                )

                learner._H = dense_to_sparse(H_carved_resized)

            # iterate
            logger.info("Running NMF")
            last_loss = np.inf
            loss_history = []
            for i in itertools.count():
                loss, loss_components = learner.iterate(PP_STRENGTH)
                dloss = abs(last_loss - loss)
                last_loss = loss
                loss_history.append(loss_components)

                if i % LOG_NMF_EVERY == 0:
                    logger.info(f"NMF iteration={i} loss={loss:.2e} dloss={dloss:.2e}")
                if dloss < DLOSS_MIN or np.sum(loss) < LOSS_MIN or i > ITER_MAX:
                    logger.info(
                        f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}"
                    )
                    break
                if i % PLOT_NMF_EVERY == 0:
                    plot.plot_nmf(learner).savefig(
                        RESULTS_DIR / f"{date}/{mix_name}/nmf-{hop_size}-{i:05d}.png"
                    )

            previous_H = learner.H
            previous_split_idx = learner.split_idx

        tick_nmf = time.time()
        results["loss"] = loss_components

        # plot NMF
        plot.plot_nmf(learner).savefig(RESULTS_DIR / f"{date}/{mix_name}/nmf.png")

        # plot loss history
        plot.plot_loss_history(loss_history).savefig(
            RESULTS_DIR / f"{date}/{mix_name}/loss.png"
        )

        # logger.info("Reconstructing tracks")
        # for i, y in enumerate(learner.reconstruct_tracks()):
        #     util.write_mp3(
        #         RESULTS_DIR / f"{date}/{mix_name}/reconstructed-{i:03d}.mp3",
        #         FS,
        #         y,
        #     )
        # logger.info("Reconstructing mix")
        # util.write_mp3(
        #     RESULTS_DIR / f"{date}/{mix_name}/reconstructed-mix.mp3",
        #     FS,
        #     learner.reconstruct_mix(),
        # )

        tick_reconstruct = time.time()

        # TODO: audio quality measure

        # get ground truth
        tau = np.arange(0, learner.V.shape[1]) * hop_size
        real_gain = mix.get_track_gain(tau)
        real_warp = mix.get_track_warp(tau)
        results["gain"] = {}
        results["warp"] = {}
        # results["gain"]["real"] = real_gain
        # results["warp"]["real"] = real_warp

        GAIN_ESTOR = param_estimator.GainEstimator.SUM
        WARP_ESTOR = param_estimator.WarpEstimator.CENTER_OF_MASS
        # estimate gain
        logger.info(f"Estimating gain with method {GAIN_ESTOR}")
        est_gain = GAIN_ESTOR.value(learner)
        # results["gain"]["est"] = est_gain
        results["gain"]["err"] = param_estimator.error(est_gain, real_gain)
        fig = plt.figure()
        plot.plot_gain(tau, est_gain, real_gain)
        fig.savefig(RESULTS_DIR / f"{date}/{mix_name}/{GAIN_ESTOR}.png")

        # estimate warp
        logger.info(f"Estimating warp with method {WARP_ESTOR}")
        est_warp = WARP_ESTOR.value(learner)
        # results["warp"]["est"] = est_warp
        results["warp"]["err"] = param_estimator.error(est_warp, real_warp)
        fig = plt.figure()
        plot.plot_warp(tau, est_warp, real_warp)
        fig.savefig(RESULTS_DIR / f"{date}/{mix_name}/{WARP_ESTOR}.png")

        # estimate high params using all estimator combos
        fill = [np.nan] * 3
        results["track_start"] = {
            "real": fill,
            "est": fill,
            "err": fill,
        }
        results["fadein_start"] = {
            "real": fill,
            "est": fill,
            "err": fill,
        }
        results["fadein_stop"] = {
            "real": fill,
            "est": fill,
            "err": fill,
        }
        results["fadeout_start"] = {
            "real": fill,
            "est": fill,
            "err": fill,
        }
        results["fadeout_stop"] = {
            "real": fill,
            "est": fill,
            "err": fill,
        }

        for i in range(3):
            real_track_start = mix.tracks[i]["start"]
            real_fadein_start = mix.tracks[i]["fadein"][0]
            real_fadein_stop = mix.tracks[i]["fadein"][1]
            real_fadeout_start = mix.tracks[i]["fadeout"][0]
            real_fadeout_stop = mix.tracks[i]["fadeout"][1]

            results["track_start"]["real"][i] = real_track_start
            results["fadein_start"]["real"][i] = real_fadein_start
            results["fadein_stop"]["real"][i] = real_fadein_stop
            results["fadeout_start"]["real"][i] = real_fadeout_start
            results["fadeout_stop"]["real"][i] = real_fadeout_stop

            logger.info(
                f"Estimating highparams for track {i} with {GAIN_ESTOR} and {WARP_ESTOR}"
            )
            (
                est_track_start,
                est_fadein_start,
                est_fadein_stop,
                est_fadeout_start,
                est_fadeout_stop,
                fig,
            ) = param_estimator.estimate_highparams(
                tau, est_gain[:, i], est_warp[:, i], plot=True
            )
            fig.savefig(
                RESULTS_DIR
                / f"{date}/{mix_name}/highparams-{i}-{GAIN_ESTOR}-{WARP_ESTOR}.png",
            )
            results["track_start"]["est"][i] = est_track_start
            results["fadein_start"]["est"][i] = est_fadein_start
            results["fadein_stop"]["est"][i] = est_fadein_stop
            results["fadeout_start"]["est"][i] = est_fadeout_start
            results["fadeout_stop"]["est"][i] = est_fadeout_stop

            results["track_start"]["err"][i] = param_estimator.error(
                est_track_start, real_track_start
            )
            results["fadein_start"]["err"][i] = param_estimator.error(
                est_fadein_start, real_track_start
            )
            results["fadein_stop"]["err"][i] = param_estimator.error(
                est_fadein_stop, real_track_start
            )
            results["fadeout_start"]["err"][i] = param_estimator.error(
                est_fadeout_start, real_track_start
            )
            results["fadeout_stop"]["err"][i] = param_estimator.error(
                est_fadeout_stop, real_track_start
            )
        tick_estimation = time.time()

        # log times
        results["times"] = {}
        results["times"]["load"] = tick_load - tick_init
        results["times"]["nmf"] = tick_nmf - tick_load
        results["times"]["reconstruction"] = tick_reconstruct - tick_nmf
        results["times"]["estimation"] = tick_estimation - tick_reconstruct
        results["times"]["total"] = tick_estimation - tick_init

        logger.info(f"Total time taken: {tick_estimation - tick_init:.2f}")

    except Exception as e:
        if e is KeyboardInterrupt:
            exit(1)
        logger.exception(f"Error when processing {mix_name}, aborting")
    finally:
        with open(RESULTS_DIR / f"{date}/{mix_name}/results.pickle", "wb") as f:
            pickle.dump(results, f)
        plt.close("all")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    # filter by no stretch no fx :) :) :)
    mixes = dict(
        filter(
            lambda i: i[1].timestretch == "none" and i[1].fx == "none",
            unmixdb.mixes.items(),
        )
    )
    logging.info(f"Will process {len(mixes)} mixes")
    if args.workers == 1:
        for mix_name, mix in mixes.items():
            worker(mix_name, mix)
    else:
        joblib.Parallel(args.workers, verbose=10)(
            joblib.delayed(worker)(mix_name, mix) for mix_name, mix in mixes.items()
        )
