import functools
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import abc

EPS = 1e-50


class Divergence(abc.ABC):
    @abc.abstractmethod
    def compute(self, V: Tensor, Vhat: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_pos(self, V: Tensor, Vhat: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_neg(self, V: Tensor, Vhat: Tensor) -> Tensor:
        raise NotImplementedError()


class Penalty(abc.ABC):
    @abc.abstractmethod
    def compute(self, X: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_neg(self, X: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def grad_pos(self, X: Tensor) -> Tensor:
        raise NotImplementedError()

def _double_backward_update(
    divergence: Divergence,
    penalties: list[tuple[Penalty, float]],
    is_param_transposed: bool,
    Vt: Tensor,
    WHt: Tensor,
    param: Parameter,
):
    param.grad = None
    # divergence
    output_neg = divergence.grad_neg(Vt.T, WHt.T).T
    output_pos = divergence.grad_pos(Vt.T, WHt.T).T

    # first backward
    WHt.backward(output_neg, retain_graph=True)
    neg = param.grad.relu_().add_(EPS)

    # second backward
    param.grad = None
    WHt.backward(output_pos)
    pos = param.grad.relu_().add_(EPS)

    # penalties
    with torch.no_grad():
        if is_param_transposed:
            for penalty, lambda_ in penalties:
                pos.add_(penalty.grad_pos(param.T).T, alpha=lambda_)
                neg.add_(penalty.grad_neg(param.T).T, alpha=lambda_)
        else:
            for penalty, lambda_ in penalties:
                pos.add_(penalty.grad_pos(param), alpha=lambda_)
                neg.add_(penalty.grad_neg(param), alpha=lambda_)

        multiplier = neg.div_(pos)
        param.data.mul_(multiplier)


class NMF(torch.nn.Module):
    def __init__(
        self,
        V: Tensor,
        W: Tensor,
        H: Tensor,
        divergence: Divergence,
        penalties_W: list[tuple[Penalty, float]],
        penalties_H: list[tuple[Penalty, float]],
        trainable_W: bool = True,
        trainable_H: bool = True,
    ):
        super().__init__()

        assert torch.all(W >= 0.0), "Tensor W should be non-negative."
        self.register_parameter(
            "W", Parameter(torch.empty(*W.size()), requires_grad=trainable_W)
        )
        self.W.data.copy_(W)
        assert torch.all(H >= 0.0), "Tensor H should be non-negative."

        self.register_parameter(
            "Ht", Parameter(torch.empty(*H.T.shape), requires_grad=trainable_H)
        )
        self.Ht.data.copy_(H.T)
        self.Vt = V.T
        self.divergence = divergence
        self.penalties_W = penalties_W
        self.penalties_H = penalties_H
        
        self._dbu_W = torch.compile(functools.partial(_double_backward_update, divergence, penalties_W, True))
        self._dbu_H = torch.compile(functools.partial(_double_backward_update, divergence, penalties_W, True))
        

    def iterate(self):
        W = self.W
        Ht = self.Ht
        Vt = self.Vt
        WHt = self.reconstruct(Ht, W)

        if self.W.requires_grad:
            WHt = self.reconstruct(Ht.detach(), W)
            self._dbu_W(Vt, WHt, W)

        if Ht.requires_grad:
            WHt = self.reconstruct(Ht, W.detach())
            self._dbu_H(Vt, WHt, Ht)
    @torch.no_grad
    def loss(self) -> tuple[float, dict]:
        losses = {}
        losses["penalties_H"] = {}
        losses["penalties_W"] = {}
        full_loss = 0

        # T, K = H.shape
        # M, _ = V.shape  # M, K

        Ht = self.Ht
        Vt = self.Vt
        W = self.W
        WHt = self.reconstruct(Ht, W)

        full_loss = self.divergence.compute(Vt.T, WHt.T) / Vt.numel()
        # assert not torch.any(torch.isnan(full_loss))
        full_loss = full_loss.item()
        losses["divergence"] = full_loss

        for penalty, lambda_ in self.penalties_H:
            loss = lambda_ * penalty.compute(Ht) / Ht.numel()
            # assert not torch.any(torch.isnan(loss))
            losses["penalties_H"][penalty.__class__.__name__] = loss.item()
            full_loss += loss.item()

        for penalty, lambda_ in self.penalties_W:
            loss = lambda_ * penalty.compute(W) / W.numel()
            # assert not torch.any(torch.isnan(loss))
            losses["penalties_W"][penalty.__class__.__name__] = loss.item()
            full_loss += loss.item()

        losses["full"] = full_loss

        return full_loss, losses

    @staticmethod
    def reconstruct(Ht, W):
        return F.linear(Ht, W)

    @property
    def V(self):
        return self.Vt.T

    @property
    def H(self):
        return self.Ht.T


##########################################################################################################
##########################################################################################################


class ItakuraSaito(Divergence):
    def compute(self, V: Tensor, Vhat: Tensor):
        return torch.sum(V / Vhat - torch.log(V / Vhat) - 1)

    def grad_pos(self, V: Tensor, Vhat: Tensor):
        # 1/Vhat
        Vhat_eps = Vhat.add(EPS)
        return Vhat_eps.reciprocal_()

    def grad_neg(self, V: Tensor, Vhat: Tensor):
        # V/Vhat**2
        return self.grad_pos(V, Vhat).square().mul_(V)


class L1(Penalty):
    def compute(self, X: Tensor):
        return X.sum().abs()

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        return torch.ones_like(X)


class L2(Penalty):
    def compute(self, X: Tensor):
        return X.pow(2).sum()

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        return 2 * X


class SmoothOverCol(Penalty):
    """
    Attempt to smooth the gain.
    $
    cal(P)_g (bold(H)) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 \
    $
    
    gradient_bold(H)^+ cal(P)_g = 4 bold(H) \
    (gradient_bold(H)^- cal(P)_g)_(i j) = 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
    
    """

    def compute(self, X: Tensor):
        return X.diff(dim=1).square().sum()

    def grad_neg(self, X: Tensor):
        T, K = X.shape
        ret = torch.zeros_like(X)
        ret[:, 1 : K - 1] = 2 * (X[:, :-2] + X[:, 2:])
        return ret

    def grad_pos(self, X: Tensor):
        return 4 * X


class SmoothDiago(Penalty):
    """
    Smooth diagonal penalty.

    The penalty is calculated as the sum of squared differences between diagonally adjacent elements in the matrix X:

        R(X) = sum_{i=1}^{T-1} sum_{j=1}^{K-1} (X_{i, j} - X_{i+1, j+1})^2

    It encourages smoothness along the diagonal of the matrix.

    Its gradient with respect to X is:

    Negative gradient (excluding edges):

        (grad_neg(R))_{i, j} = 2 * (X_{i+1, j+1} + X_{i-1, j-1})

    Positive gradient:

        (grad_pos(R)) = 4 * X
    """

    def compute(self, X: Tensor):
        return torch.sum((X[:-1, :-1] - X[1:, 1:]) ** 2)

    def grad_neg(self, X: Tensor):
        T, K = X.shape
        grad_H_neg = torch.zeros_like(X)
        grad_H_neg[1 : T - 1, 1 : K - 1] = 2 * (X[2:, 2:] + X[:-2, :-2])
        return grad_H_neg

    def grad_pos(self, X: Tensor):
        return 4 * X


class Lineness(Penalty):
    def compute(self, X: Tensor):
        T = X.shape[0]
        K = X.shape[1]

        result = 0

        for t in range(T - 2):
            for tau in range(K - 2):
                result += X[t, tau] * (
                    X[t, tau + 1] * X[t + 1, tau + 1]
                    + X[t + 1, tau] * X[t + 1, tau + 1]
                    + X[t + 1, tau] * X[t, tau + 1]
                )

        return result

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        ret = torch.zeros_like(X)

        # Extract the submatrices of X needed for the calculation
        X_i_p1_j_p1 = X[2:, 2:]  # shifted by +1 in both dims
        X_i_p1_j = X[2:, 1:-1]  # shifted by +1 in the row dim
        X_i_j_p1 = X[1:-1, 2:]  # shifted by +1 in the column dim
        X_i_m1_j_p1 = X[:-2, 2:]  # shifted by -1 in the row dim, +1 in the column dim
        X_i_m1_j = X[:-2, 1:-1]  # shifted by -1 in the row dim
        X_i_j_m1 = X[1:-1, :-2]  # shifted by -1 in the column dim
        X_i_p1_j_m1 = X[2:, :-2]  # shifted by +1 in the row dim, -1 in the column dim
        X_i_m1_j_m1 = X[:-2, :-2]  # shifted by -1 in both dims

        # Perform the main computation using the extracted submatrices
        ret[1:-1, 1:-1] = (
            X_i_j_p1 * X_i_p1_j_p1
            + X_i_p1_j * X_i_p1_j_p1
            + X_i_p1_j * X_i_j_p1
            + X_i_m1_j * X_i_j_p1
            + X_i_m1_j * X_i_m1_j_p1
            + X_i_j_m1 * X_i_p1_j
            + X_i_j_m1 * X_i_p1_j_m1
            + X_i_m1_j_m1 * X_i_m1_j
            + X_i_m1_j_m1 * X_i_j_m1
        )
        return ret


class SmoothOverRow(Penalty):
    def compute(self, X: Tensor):
        return torch.sum((X[1:, :] - X[:-1, :]) ** 2)

    def grad_neg(self, X: Tensor):
        ret = torch.zeros_like(X)
        ret[1:-1, :] = 2 * (X[:-2, :] + X[2:, :])
        return ret

    def grad_pos(self, X: Tensor):
        return 4 * X
