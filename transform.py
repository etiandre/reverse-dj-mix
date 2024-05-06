from typing import Tuple
import numpy as np

def keypoints_to_activation(keypoints: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    ret = np.zeros(shape)
    scale = keypoints.max(axis=0)

    