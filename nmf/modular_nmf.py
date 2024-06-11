import numpy as np
import scipy.sparse
import abc

EPSILON = np.finfo(np.float32).eps

SPARSE_TYPE = scipy.sparse.bsr_array


class Divergence(abc.ABC):
    @abc.abstractmethod
    def compute(self, V: np.ndarray, Vhat: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def mu_dH_num(
        self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mu_dH_dem(
        self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mu_dW_num(
        self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mu_dW_dem(
        self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray
    ) -> np.ndarray:
        pass


class Penalty(abc.ABC):
    @abc.abstractmethod
    def compute(self, X: np.ndarray | SPARSE_TYPE) -> float:
        pass

    @abc.abstractmethod
    def dX_num(self, X: np.ndarray | SPARSE_TYPE) -> np.ndarray | SPARSE_TYPE:
        pass

    @abc.abstractmethod
    def dX_dem(self, X: np.ndarray | SPARSE_TYPE) -> np.ndarray | SPARSE_TYPE:
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

    def iterate_H(self, V: np.ndarray, W: np.ndarray, H: SPARSE_TYPE) -> SPARSE_TYPE:
        Vhat = W @ H
        num = self.divergence.mu_dH_num(W, H, V, Vhat)
        dem = self.divergence.mu_dH_dem(W, H, V, Vhat)
        for penalty, lambda_ in self.penalties_H:
            num += lambda_ * penalty.dX_num(H)
            dem += lambda_ * penalty.dX_dem(H)
        dH = num / dem
        return H * dH

    def iterate_W(self, V: np.ndarray, W: np.ndarray, H: SPARSE_TYPE) -> np.ndarray:
        Vhat = W @ H
        num = self.divergence.mu_dW_num(W, H, V, Vhat)
        dem = self.divergence.mu_dW_dem(W, H, V, Vhat)
        for penalty, lambda_ in self.penalties_W:
            num += lambda_ * penalty.dX_num(W)
            dem += lambda_ * penalty.dX_dem(W)
        dW = num / dem
        return W * dW

    def loss(self, V: np.ndarray, W: np.ndarray, H: SPARSE_TYPE) -> tuple[float, dict]:
        losses = {}
        losses["penalties_H"] = {}
        losses["penalties_W"] = {}
        full_loss = 0

        Vhat = W @ H

        full_loss = self.divergence.compute(V, Vhat)
        losses["divergence"] = full_loss

        for penalty, lambda_ in self.penalties_H:
            loss = lambda_ * penalty.compute(H)
            losses["penalties_H"][str(penalty.__class__)] = loss
            full_loss += loss

        for penalty, lambda_ in self.penalties_W:
            loss = lambda_ * penalty.compute(W)
            losses["penalties_W"][str(penalty.__class__)] = loss
            full_loss += loss

        losses["full"] = full_loss

        return full_loss, losses


################### Implementations


class BetaDivergence(Divergence):
    def __init__(self, beta: float):
        self.beta = beta

    def compute(self, X: np.ndarray, Y: SPARSE_TYPE):
        if self.beta == 0:
            ret = X / Y - np.log(X / Y) - 1
        elif self.beta == 1:
            ret = X * (np.log(X) - np.log(Y)) + (Y - X)
        else:
            ret = (
                (
                    np.power(X, self.beta)
                    + (self.beta - 1) * np.power(Y, self.beta)
                    - self.beta * X * np.power(Y, self.beta - 1)
                )
                / self.beta
                / (self.beta - 1)
            )
        return float(np.mean(ret))

    def mu_dH_num(self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray):
        return W.T @ (V * Vhat ** (self.beta - 2))

    def mu_dH_dem(self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray):
        return W.T @ Vhat ** (self.beta - 1) + EPSILON

    def mu_dW_num(self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray):
        return (V * Vhat ** (self.beta - 2)) @ H.T

    def mu_dW_dem(self, W: np.ndarray, H: SPARSE_TYPE, V: np.ndarray, Vhat: np.ndarray):
        return Vhat ** (self.beta - 1) @ H.T + EPSILON


class L1(Penalty):
    def compute(self, X: np.ndarray | scipy.sparse.bsr_array) -> float:
        return np.sum(np.abs(X))

    def dX_num(self, X: np.ndarray | SPARSE_TYPE):
        return 0

    def dX_dem(self, X: np.ndarray | SPARSE_TYPE):
        return 1


class L2(Penalty):
    def compute(self, X: np.ndarray | scipy.sparse.bsr_array) -> float:
        return np.sum(X**2)

    def dX_num(self, X: np.ndarray | SPARSE_TYPE):
        return 0

    def dX_dem(self, X: np.ndarray | SPARSE_TYPE):
        return X
