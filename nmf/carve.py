from typing import Optional
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
import cv2
import numpy as np
import scipy.ndimage
import skimage


def carve(H: torch.Tensor, split_idx, threshold_dB):
    threshold = 10 ** (threshold_dB / 20)
    for left, right in zip(split_idx, split_idx[1:]):
        energy = H[left:right, :].max(dim=0)[0]
        H[left:right, energy < threshold] = 0
    return H


def carve_naive(H: torch.Tensor, threshold):
    H[H < threshold] = 0
    return H


def morpho_carve(H: torch.Tensor, threshold):
    H = H.unsqueeze(0)
    H_blur = Fv.gaussian_blur(H, 3)
    H[H_blur < threshold] = 0
    return H.squeeze(0)


def resize_cv_area(img: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    img = img.unsqueeze(0).unsqueeze(0)
    img_resized = F.interpolate(img, size=(shape[0], shape[1]), mode="area")
    img_blur = Fv.gaussian_blur(img_resized, 1)
    return img_blur.squeeze(0).squeeze(0)


def line_kernel(l: int, slope: float = 1.0):
    angle = np.arctan(slope)

    # win = np.diag(get_window(window, n, fftbins=False))
    win = np.eye(l)

    if not np.isclose(angle, np.pi / 4):
        win = scipy.ndimage.rotate(
            win, 45 - angle * 180 / np.pi, order=5, prefilter=False
        )
    win /= win.max()

    return win

import matplotlib.pyplot as plt


def resize_then_carve(
    H: torch.Tensor,
    dest_shape: tuple[int, int],
    split_idx: list[int],
    threshold: float = 1e-2,
    blur_size: int = 3,
    diag_size: int = 3,
    max_slope: float = 2,
    min_slope: Optional[float] = None,
    n_filters: int = 7,
):
    if min_slope is None:
        min_slope = 1.0 / max_slope
    elif min_slope > max_slope:
        raise ValueError(f"min_ratio={min_slope} cannot exceed max_ratio={max_slope}")

    ret = []
    for left, right in zip(split_idx, split_idx[1:]):
        Hi_np = H[left:right, :].cpu().detach().numpy()

        # TODO: angles like in librosa.segment.path_enhance
        Hi_np_enhanced = np.zeros_like(Hi_np)
        for slope in np.logspace(
            np.log2(min_slope), np.log2(max_slope), num=n_filters, base=2
        ):
            kernel = line_kernel(diag_size, slope)
            
            # TODO does not do waht its supposed to do
            np.maximum(
                Hi_np_enhanced,
                skimage.morphology.opening(Hi_np, kernel),
                out=Hi_np_enhanced,
            )
            
        Hi_np = Hi_np_enhanced

        if blur_size != 1:
            if blur_size % 2 == 0:
                blur_size += 1  # must be odd
            Hi_np = cv2.GaussianBlur(Hi_np, (blur_size, blur_size), 0)

        Hi_np[Hi_np / Hi_np.max() < threshold] = 0

        ret.append(Hi_np)

    H_np = np.concatenate(ret, axis=0)

    if dest_shape != H.shape:
        H_np = cv2.resize(
            H_np, (dest_shape[1], dest_shape[0]), interpolation=cv2.INTER_AREA
        )
    return torch.tensor(H_np).to(H.device)
