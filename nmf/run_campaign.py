from pathlib import Path
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import librosa
from pprint import pprint
import scipy.ndimage
import scipy.signal
import logging
from unmixdb import UnmixDB
import itertools
import scipy.sparse
import multiprocessing
import activation_learner, carve, plot, param_estimator
from tensorboardX import SummaryWriter


plt.style.use("dark_background")
logging.basicConfig(level=logging.INFO)
logging.getLogger("activation_learner").setLevel(logging.DEBUG)


def load_audios_parallel(paths: list[Path], fs: int):
    # def load_audio(path):
    #     return librosa.load(path, sr=fs)[0]

    # with multiprocessing.Pool() as pool:
    #     inputs = list(pool.imap(load_audio, paths))
    inputs = [librosa.load(path, sr=fs)[0] for path in paths]
    return inputs


# configuration
# =============

FS = 22050
HOP_SIZES = [5, 1]
OVERLAP_FACTOR = 4
CARVE_THRESHOLD = 1e-2
BETA = 0
NMELS = 256
# stop conditions
DLOSS_MIN = 1e-8
LOSS_MIN = -np.inf
ITER_MAX = 5000

OUTDIR = "output"

unmixdb = UnmixDB("/data2/anasynth_nonbp/schwarz/abc-dj/data/unmixdb-zenodo")

# ==============

writer = SummaryWriter()
writer.add_hparams(
    {
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
    },
    {},
)
for mix_name, mix in unmixdb.mixes.items():
    logging.info(f"Starting work on {mix_name}")
    input_paths = [
        unmixdb.refsongs[track["name"]].audio_path for track in mix.tracks
    ] + [mix.audio_path]
    pprint(input_paths)

    # load audios
    inputs = load_audios_parallel(input_paths, FS)

    # multi pass NMF
    previous_H = None
    previous_split_idx = None
    for hop_size in HOP_SIZES:
        win_size = OVERLAP_FACTOR * hop_size
        logging.info(f"Starting round with {hop_size=}s, {win_size=}s")

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
        logging.info("Running NMF")
        last_loss = np.inf
        for i in itertools.count():
            # TODO: get all components of loss
            loss = learner.iterate()
            dloss = abs(last_loss - loss)
            last_loss = loss

            writer.add_scalar("loss", loss)
            # TODO: add H to plot
            if i % 100 == 0:
                logging.info(f"NMF iteration={i} loss={loss:.2e} dloss={dloss:.2e}")
            if dloss < DLOSS_MIN or np.sum(loss) < LOSS_MIN or i > ITER_MAX:
                logging.info(f"Stopped at NMF iteration={i} loss={loss} dloss={dloss}")
                break

        # plot NMF
        writer.add_figure("nmf", plot.plot_nmf(learner))

        previous_H = learner.H
        previous_split_idx = learner.split_idx

    # for i in range(len(mix.tracks)):
    #     logging.info(f"Reconstructing track {i}")
    #     reconstructed_audio = learner.reconstruct(i)
    #     soundfile.write(
    #         f"{mix_name}-reconstructed-{i:03d}.ogg", reconstructed_audio, samplerate=FS
    #     )

    # TODO: audio quality measure

    # parameter estimation
    T = np.arange(0, int(mix.duration / hop_size * FS), int(hop_size * FS)) / FS
    real_volumes = mix.get_track_volumes(T)
    real_timeremap = mix.get_track_positions(T)

    for method in param_estimator.VolumeEstimator:
        logging.info(f"Estimating volume with method {method}")
        est_volumes = method.value(learner)
        rel_error = param_estimator.rel_error(est_volumes, real_volumes)
        writer.add_scalar(f"volume_error/{mix_name}-{method}", rel_error)

        fig = plt.figure()
        plot.plot_volume(est_volumes, learner.hop_size, real_volumes)
        writer.add_figure(f"volume/{mix_name}-{method}", fig)

    for method in param_estimator.TimeRemappingEstimator:
        logging.info(f"Estimating timeremap with method {method}")
        est_timeremap = method.value(learner)
        rel_error = param_estimator.rel_error(est_timeremap, real_timeremap)
        writer.add_scalar(f"volume_error/{mix_name}-{method}", rel_error)

        fig = plt.figure()
        plot.plot_timeremap(est_timeremap, hop_size, real_timeremap)
        writer.add_figure(f"timeremap/{mix_name}-{method}", fig)

        # TODO: save stuff

    break
