from pathlib import Path
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import librosa
from pprint import pprint
import scipy.ndimage
import scipy.signal
import logging
import itertools
import scipy.sparse
import datetime
import os
import pickle
import time

from unmixdb import UnmixDB
from abcdj import ABCDJ
import activation_learner, carve, plot, param_estimator


# plt.style.use("dark_background")
date = datetime.datetime.now().isoformat()
os.makedirs(f"results/{date}")

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler(f"results/{date}/output.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def load_audios(paths: list[Path], fs: int):
    # def load_audio(path):
    #     return librosa.load(path, sr=fs)[0]

    # with multiprocessing.Pool() as pool:
    #     inputs = list(pool.imap(load_audio, paths))
    inputs = [librosa.load(path, sr=fs)[0] for path in paths]
    return inputs


# configuration
# =============

FS = 22050
HOP_SIZES = [5.0, 1.0]
OVERLAP_FACTOR = 4
CARVE_THRESHOLD = 1e-2
BETA = 0
NMELS = 256
VOL_FILTER_SIZE = 0.1
# stop conditions
DLOSS_MIN = 1e-8
LOSS_MIN = -np.inf
ITER_MAX = 4000

OUTDIR = "output"

unmixdb = UnmixDB("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")
# unmixdb = UnmixDB("/home/etiandre/stage/datasets/unmixdb-zenodo")
abcdj = ABCDJ(
    "/data2/anasynth_nonbp/schwarz/abc-dj/src-git/unmixing/results-unmixdb-full"
)

# ==============
logging.info(f"{len(unmixdb.mixes)=}")
for mix_name, mix in unmixdb.mixes.items():
    results = {}
    try:
        os.makedirs(f"results/{date}/{mix_name}")
        results["hyperparams"] = {
            "FS": FS,
            "ROUNDS": len(HOP_SIZES),
            "HOP_SIZES": repr(HOP_SIZES),
            "OVERLAP_FACTOR": OVERLAP_FACTOR,
            "CARVE_THRESHOLD": CARVE_THRESHOLD,
            "DLOSS_MIN": DLOSS_MIN,
            "LOSS_MIN": LOSS_MIN,
            "ITER_MAX": ITER_MAX,
            "BETA": BETA,
            "NMELS": NMELS,
            "VOL_FILTER_SIZE": VOL_FILTER_SIZE,
        }
        logger.info(f"Starting work on {mix_name}")
        input_paths = [
            unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks
        ] + [mix.audio_path]
        pprint(input_paths)

        tick_init = time.time()

        # load audios
        inputs = load_audios(input_paths, FS)

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
                beta=BETA,
                win_size=win_size,
                hop_size=hop_size,
            )

            # carve and resize H from previous round
            if previous_H is not None:
                H_carved = carve.carve(previous_H, previous_split_idx, CARVE_THRESHOLD)
                H_carved_resized = carve.resize_cv_area(H_carved, learner.H.shape)
                plot.plot_carve_resize(H_carved, H_carved_resized)
                learner.nmf.H = scipy.sparse.bsr_array(H_carved_resized)

            # iterate
            logger.info("Running NMF")
            last_loss = np.inf
            for i in itertools.count():
                # TODO: get all components of loss
                loss = learner.iterate()
                dloss = abs(last_loss - loss)
                last_loss = loss

                # TODO: add H to plot
                if i % 100 == 0:
                    logger.info(f"NMF iteration={i} loss={loss:.2e} dloss={dloss:.2e}")
                if dloss < DLOSS_MIN or np.sum(loss) < LOSS_MIN or i > ITER_MAX:
                    logger.info(
                        f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}"
                    )
                    break

            previous_H = learner.H
            previous_split_idx = learner.split_idx

        tick_nmf = time.time()

        # plot NMF
        plot.plot_nmf(learner).savefig(f"results/{date}/{mix_name}/nmf.png")
        results["loss"] = loss

        # for i in range(len(mix.tracks)):
        #     logger.info(f"Reconstructing track {i}")
        #     reconstructed_audio = learner.reconstruct(i)
        #     soundfile.write(
        #         f"{mix_name}-reconstructed-{i:03d}.ogg", reconstructed_audio, samplerate=FS
        #     )

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
            fig.savefig(f"results/{date}/{mix_name}/{estimator}.png")
            results["est_gain"][str(estimator)] = est_gain

        # estimate warp
        results["est_warp"] = {}
        for estimator in param_estimator.WarpEstimator:
            logger.info(f"Estimating warp with method {estimator}")
            est_warp = estimator.value(learner)

            fig = plt.figure()
            plot.plot_warp(tau, est_warp, real_warp)
            fig.savefig(f"results/{date}/{mix_name}/{estimator}.png")

            results["est_warp"][str(estimator)] = est_warp

        # estimate high params using all estimator combos
        results["highparams"] = [{}, {}, {}]
        for i in range(3):
            ret = {}
            ret["real_track_start"] = mix.tracks[i]["start"]
            ret["real_fadein_start"] = mix.tracks[i]["fadein"][0]
            ret["real_fadein_stop"] = mix.tracks[i]["fadein"][1]
            ret["real_fadeout_start"] = mix.tracks[i]["fadeout"][0]
            ret["real_fadeout_stop"] = mix.tracks[i]["fadeout"][1]
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
                    est_fadein_slope,
                    est_fadeout_start,
                    est_fadeout_stop,
                    est_fadeout_slope,
                    fig,
                ) = param_estimator.estimate_highparams(
                    tau, est_gain[:, i], est_warp[:, i], plot=True
                )
                fig.savefig(
                    f"results/{date}/{mix_name}/highparams-{i}-{g_estor}-{w_estor}.png",
                )
                ret[str(g_estor)] = {}
                ret[str(g_estor)][str(w_estor)] = {}
                ret[str(g_estor)][str(w_estor)]["est_track_start"] = est_track_start
                ret[str(g_estor)][str(w_estor)]["est_fadein_start"] = est_fadein_start
                ret[str(g_estor)][str(w_estor)]["est_fadein_stop"] = est_fadein_stop
                ret[str(g_estor)][str(w_estor)]["est_fadein_slope"] = est_fadein_slope
                ret[str(g_estor)][str(w_estor)]["est_fadeout_start"] = est_fadeout_start
                ret[str(g_estor)][str(w_estor)]["est_fadeout_stop"] = est_fadeout_stop
                ret[str(g_estor)][str(w_estor)]["est_fadeout_slope"] = est_fadeout_slope
                results["highparams"][i] = ret

        tick_estimation = time.time()

        # log times
        results["times"] = {}
        results["times"]["load"] = tick_load - tick_init
        results["times"]["nmf"] = tick_nmf - tick_load
        results["times"]["estimation"] = tick_estimation - tick_nmf
        results["times"]["total"] = tick_estimation - tick_init

        logger.info(f"Total time taken: {tick_estimation - tick_init:.2f}")

    except Exception:
        logger.exception(f"Error when processing {mix_name}, aborting")
    finally:
        with open(f"results/{date}/{mix_name}/results.pickle", "wb") as f:
            pickle.dump(results, f)
        plt.close("all")
