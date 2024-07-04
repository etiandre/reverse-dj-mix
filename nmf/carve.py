import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv


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


def resize_then_carve(
    H: torch.Tensor,
    dest_shape: tuple[int, int],
    threshold: float,
    n: int,
    resize_mode="area",
):
    H = H.unsqueeze(0).unsqueeze(0)
    H = F.interpolate(H, size=(dest_shape[0], dest_shape[1]), mode=resize_mode)
    if n != 1:
        H = Fv.gaussian_blur(H, n)
    H[H < threshold] = 0

    # TODO: remove islands (by erosion-dilatation then masking)
    
    return H.squeeze(0).squeeze(0)
