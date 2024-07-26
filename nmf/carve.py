from typing import Optional
import warnings
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
import cv2
import numpy as np
import scipy.ndimage
import skimage
import matplotlib.pyplot as plt
import librosa
import logging
import plot
import scipy.signal

logger = logging.getLogger(__name__)


def line_kernel(
    window: str,
    n: int,
    *,
    slope: float = 1.0,
    angle: Optional[float] = None,
) -> np.ndarray:
    if angle is None:
        angle = np.arctan(slope)

    win = np.diag(scipy.signal.get_window(window, n, fftbins=False))

    if not np.isclose(angle, np.pi / 4):
        win = scipy.ndimage.rotate(
            win, 45 - angle * 180 / np.pi, order=5, prefilter=False
        )

    np.clip(win, 0, None, out=win)
    win /= win.max()

    return win


def one_pixel_line_kernel(n, slope):
    angle = np.arctan(slope)
    x = int(n * np.cos(angle))
    y = int(n * np.sin(angle))
    ker = np.zeros((x + 1, y + 1)).astype(float)
    rr, cc = skimage.draw.line(0, 0, x, y)
    ker[rr, cc] = 1.0
    return ker


def line_enhance(
    H: np.ndarray,
    split_idx: list[int],
    size: int,
    max_slope: float,
    n_filters: int,
    diag_window: str,
):
    min_slope = 1.0 / max_slope
    ret = np.zeros_like(H)
    for left, right in zip(split_idx, split_idx[1:]):
        Hi = H[left:right, :]
        # Hi = np.ascontiguousarray(Hi)
        # Hi_skel = skimage.morphology.skeletonize(Hi).astype(float)

        for slope in np.logspace(
            np.log2(min_slope), np.log2(max_slope), num=n_filters, base=2
        ):
            kernel = one_pixel_line_kernel(n=size, slope=slope)
            np.maximum(
                ret[left:right, :],
                skimage.morphology.opening(Hi_skel, kernel),
                out=ret[left:right, :],
            )
    return ret


def blur(
    H: np.ndarray,
    split_idx: list[int],
    size: int = 3,
):
    if size == 1:
        return H
    ret = []
    for left, right in zip(split_idx, split_idx[1:]):
        Hi = H[left:right, :]
        if size % 2 == 0:
            size += 1  # must be odd
        Hi_blur = cv2.GaussianBlur(Hi, (size, size), 0)
        Hi_blur *= Hi.max() / Hi_blur.max()
        ret.append(Hi_blur)

    return np.concatenate(ret, axis=0)


def H_interpass_enhance(
    H: torch.Tensor,
    dest_shape: tuple[int, int],
    split_idx: list[int],
    threshold: float = 1e-2,
    blur_size: int = 3,
    diag_size: int = 3,
    max_slope: float = 2,
    n_filters: int = 7,
    diag_window: str = "boxcar",
    doplot=False,
):
    logger.info(
        f"H_enhance: {dest_shape=}, {threshold=}, {blur_size=}, {diag_size=}, {max_slope=}, {n_filters=}, {diag_window=}"
    )

    H_np = H.detach().cpu().numpy()

    # filter for lines
    if diag_size <= 3:
        logger.warn(f"diag size ({diag_size}) is too small, skipping line enhance")
        H_line = H_np
    else:
        H_line = line_enhance(
            H_np, split_idx, diag_size, max_slope, n_filters, diag_window
        )

    # blur
    H_blur = blur(H_line, split_idx, blur_size)

    # threshold
    H_blur[H_blur / H_blur.max() < threshold] = 0

    # resize
    if dest_shape != H.shape:
        H_resized = cv2.resize(
            H_blur, (dest_shape[1], dest_shape[0]), interpolation=cv2.INTER_AREA
        )
    else:
        H_resized = H_blur

    if doplot:
        fig, axs = plt.subplots(1, 3)
        plot.plot_H(H_np, ax=axs[0])
    return torch.tensor(H_resized).to(H.device)
