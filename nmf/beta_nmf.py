import numpy as np
import scipy.sparse

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


def smooth_penalty(lambda_):
    return lambda_ * (np.sum(R.flatten() - np.log(R.flatten())) - K * (N - 1))


def mu_H(W, H, V, beta, l1_reg, l2_reg):
    Vhat = W @ H
    num = W.T @ (V * Vhat ** (beta - 2))
    dem = W.T @ Vhat ** (beta - 1) + EPSILON

    # Add L1 and L2 regularization
    if l1_reg > 0:
        dem += l1_reg
    if l2_reg > 0:
        dem += l2_reg * H

    dH = num / dem
    return H * dH


def mu_H_smooth_diago(W, H, V, beta, lambda_):
    H = H.toarray()
    Vhat = W @ H
    num = W.T @ (V * Vhat ** (beta - 2))
    dem = W.T @ Vhat ** (beta - 1) + EPSILON

    grad_H_pos = 4 * H
    grad_H_neg = np.zeros_like(H)
    for i in range(1, H.shape[0] - 1):
        for j in range(1, H.shape[1] - 1):
            grad_H_neg[i, j] = 2 * (H[i + 1, j + 1] - H[i - 1, j - 1])

    num += lambda_ * grad_H_neg
    dem += lambda_ * grad_H_pos
    dH = num / dem
    return scipy.sparse.bsr_array(H * dH)


def mu_H_variance_col(W, H, V, beta, lambda_):
    H = H.toarray()
    Vhat = W @ H
    num = W.T @ (V * Vhat ** (beta - 2))
    dem = W.T @ Vhat ** (beta - 1) + EPSILON

    F, N = H.shape
    grad_H_pos = 2 / (N * F) * H
    grad_H_neg = np.zeros_like(H)
    for i in range(0, H.shape[0] - 1):
        for j in range(0, H.shape[1] - 1):
            grad_H_neg[i, j] = 2 / (F * N**2) * np.sum(H[i, :])

    num += lambda_ * grad_H_neg
    dem += lambda_ * grad_H_pos
    dH = num / dem
    return scipy.sparse.bsr_array(H * dH)


def mu_H_smooth_fevotte(W, H, V, lambda_):
    H = H.toarray()
    H[H == 0] = EPSILON
    F, N = V.shape

    Vhat = W @ H
    # Update H
    dem = W.T @ (Vhat**-1) + EPSILON
    num = W.T @ (V * Vhat**-2)
    Ht = H.copy()

    # first column: n = 0
    p2 = dem[:, 0] + lambda_ / H[:, 1]
    p1 = -lambda_
    p0 = -num[:, 0] * Ht[:, 0] ** 2
    H[:, 0] = (np.sqrt(p1**2 - 4 * p2 * p0) - p1) / (2 * p2)

    # middle columns: n = 1 to N-2
    for n in range(1, N - 1):
        H[:, n] = np.sqrt(
            (num[:, n] * Ht[:, n] ** 2 + lambda_ * H[:, n - 1])
            / (dem[:, n] + lambda_ / H[:, n + 1])
        )

    # last column: n = N-1
    p2 = dem[:, N - 1]
    p1 = lambda_
    p0 = -(num[:, N - 1] * Ht[:, N - 1] ** 2 + lambda_ * H[:, N - 2])
    H[:, N - 1] = (np.sqrt(p1**2 - 4 * p2 * p0) - p1) / (2 * p2)

    return scipy.sparse.bsr_array(H)


def mu_W(W, H, V, beta, l1_reg, l2_reg):
    Vhat = W @ H
    num = (V * Vhat ** (beta - 2)) @ H.T
    dem = Vhat ** (beta - 1) @ H.T + EPSILON

    # Add L1 and L2 regularization
    if l1_reg > 0:
        dem += l1_reg
    if l2_reg > 0:
        dem += l2_reg * W

    dW = num / dem
    return W * dW


class BetaNMF:
    def __init__(
        self,
        V: np.ndarray,
        W: np.ndarray,
        H: scipy.sparse.sparray,
        beta: float,
        fixed_W: bool = False,
        fixed_H: bool = False,
        alpha_W: float = 0,
        alpha_H: float = 0,
        l1_ratio: float = 0,
        lambda_smooth: float = 0,
        lambda_diago: float = 0,
        lambda_variance: float = 0,
    ):
        assert not np.isnan(H.toarray()).any(), "H contains NaN"
        assert np.all(H.toarray() >= 0), "H is negative"
        # if beta == 0:
        #     assert np.all(H > 0), "H cannot have 0 values if beta==0"
        assert not np.isnan(W).any(), "W contains NaN"
        assert np.all(W >= 0), "W is negative"
        assert not np.isnan(V).any(), "V contains NaN"
        assert np.all(V >= 0), "V is negative"
        assert not (fixed_W and fixed_H), "Cannot fix both W and H"

        if lambda_smooth > 0:
            assert (
                alpha_W == 0 and alpha_H == 0
            ), "Cannot use l1 or l2 regularization with is_smooth"
            assert beta == 0, "Cannot use beta!=0 with is_smooth"
            algorithm = "is_smooth_fevotte"
        elif lambda_diago > 0:
            algorithm = "mu_smooth_diago"
        elif lambda_variance > 0:
            algorithm = "mu_variance"
        else:
            algorithm = "mu"
        print(f"Using algorithm {algorithm}")
        self.V = V  # FxN
        self.W = W  # FxK
        self.H = H  # KxN
        self.beta = beta
        self.fixed_W = fixed_W
        self.fixed_H = fixed_H
        self.lambda_smooth = lambda_smooth
        self.lambda_diago = lambda_diago
        self.lambda_variance = lambda_variance
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
            elif self.algorithm == "is_smooth_fevotte":
                self.H = mu_H_smooth_fevotte(self.W, self.H, self.V, self.lambda_smooth)
            elif self.algorithm == "mu_smooth_diago":
                self.H = mu_H_smooth_diago(
                    self.W, self.H, self.V, self.beta, self.lambda_diago
                )
            elif self.algorithm == "mu_variance":
                self.H = mu_H_variance_col(
                    self.W, self.H, self.V, self.beta, self.lambda_variance
                )
            else:
                raise NotImplementedError

        if not self.fixed_W:
            self.W = mu_W(
                self.W, self.H, self.V, self.beta, self.l1_reg_W, self.l2_reg_W
            )

    def loss(self):
        # if self.lambda_variance > 0:
        #     losses.append(
        #         self.lambda_variance * np.mean(np.var(self.H.toarray(), axis=0))
        #     )
        # TODO: add all penalties
        return beta_divergence(self.V, self.W @ self.H, self.beta)
