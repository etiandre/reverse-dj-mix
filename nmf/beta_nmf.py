import numpy as np

EPSILON = np.finfo(np.float32).eps


def beta_divergence(x: np.ndarray, y: np.ndarray, beta: float):
    """returns beta divergence

    Args:
        a (np.ndarray): first matrix
        b (np.ndarray): second matrix
        beta (float): beta parameter
    """

    if beta == 0:
        ret = x / y - np.log(x / y) - 1
    elif beta == 1:
        ret = x * (np.log(x) - np.log(y)) + (y - x)
    else:
        ret = (
            (
                np.power(x, beta)
                + (beta - 1) * np.power(y, beta)
                - beta * x * np.power(y, beta - 1)
            )
            / beta
            / (beta - 1)
        )
    return np.mean(ret)


def mu_H(W, H, V, beta, l1_reg_H, l2_reg_H):
    Vhat = W @ H
    num = W.T @ (V * Vhat ** (beta - 2))
    dem = W.T @ Vhat ** (beta - 1) + EPSILON

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        dem += l1_reg_H
    if l2_reg_H > 0:
        dem += l2_reg_H * H

    dH = num / dem
    return H * dH


def mu_W(W, H, V, beta, l1_reg_W, l2_reg_W):
    Vhat = W @ H
    num = (V * Vhat ** (beta - 2)) @ H.T
    dem = Vhat ** (beta - 1) @ H.T + EPSILON

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        dem += l1_reg_W
    if l2_reg_W > 0:
        dem += l2_reg_W * W

    dW = num / dem
    return W * dW


class BetaNMF:
    def __init__(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        beta: float,
        fixed_W: bool = False,
        fixed_H: bool = False,
        algorithm="mu",
        alpha_W: float = 0,
        alpha_H: float = 0,
        l1_ratio: float = 0,
    ):
        assert not np.isnan(H).any(), "H contains NaN"
        assert np.all(H >= 0), "H is negative"
        # if beta == 0:
        #     assert np.all(H > 0), "H cannot have 0 values if beta==0"
        assert not np.isnan(W).any(), "W contains NaN"
        assert np.all(W >= 0), "W is negative"
        assert not np.isnan(V).any(), "V contains NaN"
        assert np.all(V >= 0), "V is negative"
        assert not (fixed_W and fixed_H), "Cannot fix both W and H"
        assert algorithm in ["mu"]

        self.V = V  # FxN
        self.W = W  # FxK
        self.H = H  # KxN
        self.beta = beta
        self.fixed_W = fixed_W
        self.fixed_H = fixed_H
        self.algorithm = algorithm

        # compute regularization
        F, N = V.shape
        self.l1_reg_W = N * alpha_W * l1_ratio
        self.l1_reg_H = F * alpha_H * l1_ratio
        self.l2_reg_W = N * alpha_W * (1.0 - l1_ratio)
        self.l2_reg_H = F * alpha_H * (1.0 - l1_ratio)

    def iterate(self):
        if not self.fixed_H:
            if self.algorithm == "mu":
                self.H = mu_H(
                    self.W, self.H, self.V, self.beta, self.l1_reg_H, self.l2_reg_H
                )
            else:
                raise NotImplementedError

        if not self.fixed_W:
            if self.algorithm == "mu":
                self.W = mu_W(
                    self.W, self.H, self.V, self.beta, self.l1_reg_W, self.l2_reg_W
                )
            else:
                raise NotImplementedError

    def loss(self):
        loss = beta_divergence(self.V, self.W @ self.H, self.beta)
        return loss
