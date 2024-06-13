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
    # (modular_nmf.L1(), 9643.097400544748),
    # (modular_nmf.SmoothGain(), 1572.5437219589044)
    # (modular_nmf.VirtanenTemporalContinuity(), 1)
]

# stop conditions
DLOSS_MIN = -np.inf
LOSS_MIN = -np.inf
ITER_MAX = 3000

## other stuff
# logging
PLOT_NMF_EVERY = 500
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
        pprint(input_paths)

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
                loss, loss_components = learner.iterate()
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
        results["H"] = learner.H
        results["loss"] = loss_components
        results["loss_history"] = loss_history

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
        results["real_gain"] = real_gain
        results["real_warp"] = real_warp

        # estimate gain
        results["est_gain"] = {}
        for estimator in param_estimator.GainEstimator:
            logger.info(f"Estimating gain with method {estimator}")
            est_gain = estimator.value(learner)

            fig = plt.figure()
            plot.plot_gain(tau, est_gain, real_gain)
            fig.savefig(RESULTS_DIR / f"{date}/{mix_name}/{estimator}.png")
            results["est_gain"][str(estimator)] = est_gain

        # estimate warp
        results["est_warp"] = {}
        for estimator in param_estimator.WarpEstimator:
            logger.info(f"Estimating warp with method {estimator}")
            est_warp = estimator.value(learner)

            fig = plt.figure()
            plot.plot_warp(tau, est_warp, real_warp)
            fig.savefig(RESULTS_DIR / f"{date}/{mix_name}/{estimator}.png")

            results["est_warp"][str(estimator)] = est_warp

        # estimate high params using all estimator combos
        results["highparams"] = [{}, {}, {}]
        for i in range(3):
            ret = {}
            ret["real"] = {}
            ret["real"]["track_start"] = mix.tracks[i]["start"]
            ret["real"]["fadein_start"] = mix.tracks[i]["fadein"][0]
            ret["real"]["fadein_stop"] = mix.tracks[i]["fadein"][1]
            ret["real"]["fadeout_start"] = mix.tracks[i]["fadeout"][0]
            ret["real"]["fadeout_stop"] = mix.tracks[i]["fadeout"][1]

            ret["est"] = {}
            for (g_estor, est_gain), (
                w_estor,
                est_warp,
            ) in itertools.product(
                results["est_gain"].items(), results["est_warp"].items()
            ):
                logger.info(
                    f"Estimating highparams for track {i} with {g_estor} and {w_estor}"
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
                    / f"{date}/{mix_name}/highparams-{i}-{g_estor}-{w_estor}.png",
                )
                estor = str(g_estor) + " " + str(w_estor)
                ret["est"][estor] = {}
                ret["est"][estor]["track_start"] = est_track_start
                ret["est"][estor]["fadein_start"] = est_fadein_start
                ret["est"][estor]["fadein_stop"] = est_fadein_stop
                ret["est"][estor]["fadeout_start"] = est_fadeout_start
                ret["est"][estor]["fadeout_stop"] = est_fadeout_stop
                results["highparams"][i].update(ret)

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
