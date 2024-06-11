import abc

import numpy as np
from numba import jit
from common import ArrayType, dense_to_sparse

EPSILON = np.finfo(np.float32).eps


class Divergence(abc.ABC):
    @abc.abstractmethod
    def compute(self, V: ArrayType, Vhat: ArrayType) -> float:
        pass

    @abc.abstractmethod
    def mu_dH_num(
        self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType
    ) -> ArrayType | float:
        pass

    @abc.abstractmethod
    def mu_dH_dem(
        self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType
    ) -> ArrayType | float:
        pass

    @abc.abstractmethod
    def mu_dW_num(
        self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType
    ) -> ArrayType | float:
        pass

    @abc.abstractmethod
    def mu_dW_dem(
        self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType
    ) -> ArrayType | float:
        pass


class Penalty(abc.ABC):
    @abc.abstractmethod
    def compute(self, X: ArrayType | ArrayType) -> float:
        pass

    @abc.abstractmethod
    def grad_neg(self, X: ArrayType | ArrayType) -> ArrayType | float:
        pass

    @abc.abstractmethod
    def grad_pos(self, X: ArrayType | ArrayType) -> ArrayType | float:
        pass


class NMF:
    def __init__(
        self,
        divergence: Divergence,
        penalties_W: list[tuple[Penalty, float]],
        penalties_H: list[tuple[Penalty, float]],
    ):
        self.divergence = divergence
        self.penalties_W = penalties_W
        self.penalties_H = penalties_H

    def iterate_H(self, V: ArrayType, W: ArrayType, H: ArrayType) -> ArrayType:
        Vhat = W @ H
        num = self.divergence.mu_dH_num(W, H, V, Vhat)
        dem = self.divergence.mu_dH_dem(W, H, V, Vhat)
        assert not np.any(np.isnan(num))
        assert not np.any(np.isnan(dem))
        assert np.all(num >= 0)
        assert np.all(dem > 0)
        for penalty, lambda_ in self.penalties_H:
            num += lambda_ * penalty.grad_neg(H)
            dem += lambda_ * penalty.grad_pos(H)
            assert not np.any(np.isnan(num))
            assert not np.any(np.isnan(dem))
            assert np.all(num >= 0)
            assert np.all(dem > 0)
        dH = num / dem
        assert not np.any(np.isnan(dH))
        assert np.all(dH >= 0)
        return H * dH

    def iterate_W(self, V: ArrayType, W: ArrayType, H: ArrayType) -> ArrayType:
        Vhat = W @ H
        num = self.divergence.mu_dW_num(W, H, V, Vhat)
        dem = self.divergence.mu_dW_dem(W, H, V, Vhat)
        assert not np.any(np.isnan(num))
        assert not np.any(np.isnan(dem))
        assert np.all(num >= 0)
        assert np.all(dem > 0)
        for penalty, lambda_ in self.penalties_W:
            num += lambda_ * penalty.grad_neg(W)
            dem += lambda_ * penalty.grad_pos(W)
            assert not np.any(np.isnan(num))
            assert not np.any(np.isnan(dem))
            assert np.all(num >= 0)
            assert np.all(dem > 0)
        dW = num / dem
        assert not np.any(np.isnan(dW))
        assert np.all(dW >= 0)
        return W * dW

    def loss(self, V: ArrayType, W: ArrayType, H: ArrayType) -> tuple[float, dict]:
        losses = {}
        losses["penalties_H"] = {}
        losses["penalties_W"] = {}
        full_loss = 0

        Vhat = W @ H

        full_loss = self.divergence.compute(V, Vhat)
        assert not np.any(np.isnan(full_loss))
        losses["divergence"] = full_loss

        for penalty, lambda_ in self.penalties_H:
            loss = lambda_ * penalty.compute(H)
            assert not np.any(np.isnan(loss))
            losses["penalties_H"][str(penalty.__class__)] = loss
            full_loss += loss

        for penalty, lambda_ in self.penalties_W:
            loss = lambda_ * penalty.compute(W)
            assert not np.any(np.isnan(loss))
            losses["penalties_W"][str(penalty.__class__)] = loss
            full_loss += loss

        losses["full"] = full_loss

        return full_loss, losses


################### Implementations


class BetaDivergence(Divergence):
    def __init__(self, beta: float):
        self.beta = beta

    def compute(self, V: ArrayType, Vhat: ArrayType):
        if self.beta == 0:
            ret = V / Vhat - np.log(V / Vhat) - 1
        elif self.beta == 1:
            ret = V * (np.log(V) - np.log(Vhat)) + (Vhat - V)
        else:
            ret = (
                (
                    np.power(V, self.beta)
                    + (self.beta - 1) * np.power(Vhat, self.beta)
                    - self.beta * V * np.power(Vhat, self.beta - 1)
                )
                / self.beta
                / (self.beta - 1)
            )
        return float(np.mean(ret))

    def mu_dH_num(self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType):
        return W.T @ (V * Vhat ** (self.beta - 2))

    def mu_dH_dem(self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType):
        return W.T @ Vhat ** (self.beta - 1) + EPSILON

    def mu_dW_num(self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType):
        return (V * Vhat ** (self.beta - 2)) @ H.T

    def mu_dW_dem(self, W: ArrayType, H: ArrayType, V: ArrayType, Vhat: ArrayType):
        return Vhat ** (self.beta - 1) @ H.T + EPSILON


class L1(Penalty):
    """
    L1-norm regularization.

    The L1-norm regularization is given by the formula:

        R(X) = ||X||_1 = sum(|X_ij|)

    where ||X||_1 is the L1-norm of the matrix X, and |X_ij| is the absolute value of the element at
    (i, j) in the matrix X.

    This regularization encourages sparsity in the solution by penalizing the absolute values of the elements.

    Since the function is non-differentiable at zero, its sub-gradient can be used instead:

        dR/dX = sign(X)
    """

    def compute(self, X: ArrayType) -> float:
        return np.sum(np.abs(X))

    def grad_neg(self, X: ArrayType):
        return 0

    def grad_pos(self, X: ArrayType):
        return 1


class L2(Penalty):
    """
    L2-norm regularization.

    The L2-norm regularization is given by the formula:

        R(X) = ||X||_2^2 = sum(X_ij^2)

    Its gradient with respect to X is:

        dR/dX = 2 * X
    """

    def compute(self, X: ArrayType):
        return float(np.sum(X**2))

    def grad_neg(self, X: ArrayType):
        return 0

    def grad_pos(self, X: ArrayType):
        return X


class FevotteSmooth(Penalty):
    """
    Fevotte, Cedric. « Majorization-Minimization Algorithm for Smooth Itakura-Saito
    Nonnegative Matrix Factorization ». In 2011 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP), 1980-83. Prague, Czech Republic:
    IEEE, 2011. https://doi.org/10.1109/ICASSP.2011.5946898.
    """

    def compute(self, X: ArrayType) -> float:
        raise NotImplementedError

    def grad_neg(self, X: ArrayType):
        raise NotImplementedError

    def grad_pos(self, X: ArrayType):
        raise NotImplementedError


class SmoothGain(Penalty):
    """
    Attempt to smooth the gain.
    $
    cal(P)_g (bold(H)) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 \
    $
    
    gradient_bold(H)^+ cal(P)_g = 4 bold(H) \
    (gradient_bold(H)^- cal(P)_g)_(i j) = 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
    
    """

    def compute(self, X: ArrayType) -> float:
        return float(np.mean(np.diff(X, axis=1) ** 2))

    def grad_neg(self, X: ArrayType):
        ret = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(1, X.shape[1] - 1):
                ret[i, j] = 2 * (X[i, j - 1] + X[i, j + 1])

        return dense_to_sparse(ret / (X.shape[0] * X.shape[1] - 1))

    def grad_pos(self, X: ArrayType):
        return 4 / (X.shape[0] * X.shape[1] - 1) * X


class VirtanenTemporalContinuity(Penalty):
    """
    Monaural Sound Source Separation by Nonnegative Matrix Factorization With Temporal
    Continuity and Sparseness Criteria
    Virtanen et Al, 2007
    """

    def compute(self, X: ArrayType):
        T, K = X.shape
        std = np.std(X, axis=1)
        diffs = np.diff(X, axis=1)
        first_sum = np.sum(diffs**2, axis=1)
        second_sum = np.sum(1 / std**2 * first_sum)
        return second_sum / T / K

    def grad_neg(self, X: ArrayType):
        T, K = X.shape
        ret = np.zeros_like(X)
        sums_sq = np.sum(X**2, axis=1)
        diffs = X[:, 1:] - X[:, :-1]

        for t in range(T):
            for tau in range(1, K - 1):
                ret[t, tau] = 2 * K / sums_sq[t] * (
                    X[t, tau - 1] + X[t, tau + 1]
                ) + 2 * T / sums_sq[t] ** 2 * X[t, tau] * np.sum(diffs[t] ** 2)
        return dense_to_sparse(ret)

    def grad_pos(self, X: ArrayType):
        T, K = X.shape
        ret = np.zeros_like(X)
        sums_sq = np.sum(X**2, axis=1)
        for t in range(T):
            for tau in range(1, K - 1):
                ret[t, tau] = X[t, tau] / sums_sq[t]
        return dense_to_sparse(4 * K * ret)
