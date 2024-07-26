import functools
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import abc

EPS = 1e-15
COMPILE = False


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
    penalties: list[Penalty],
    is_param_transposed: bool,
    penalties_lambdas: list[float],
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

    param.grad = None

    # penalties
    with torch.no_grad():
        if is_param_transposed:
            for penalty, lambda_ in zip(penalties, penalties_lambdas):
                if lambda_ == 0:
                    continue
                pos.add_(penalty.grad_pos(param.T).T, alpha=lambda_)
                neg.add_(penalty.grad_neg(param.T).T, alpha=lambda_)
        else:
            for penalty, lambda_ in zip(penalties, penalties_lambdas):
                if lambda_ == 0:
                    continue
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
        penalties_W: list[Penalty],
        penalties_H: list[Penalty],
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

        # In [2]: torch._dynamo.list_backends()
        # Out[2]: ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
        if COMPILE:
            self._dbu_W = torch.compile(
                functools.partial(
                    _double_backward_update, divergence, penalties_W, False
                ),
            )
            self._dbu_H = torch.compile(
                functools.partial(
                    _double_backward_update, divergence, penalties_H, True
                ),
            )
        else:
            self._dbu_W = functools.partial(
                _double_backward_update, divergence, penalties_W, False
            )
            self._dbu_H = functools.partial(
                _double_backward_update, divergence, penalties_H, True
            )

    def iterate(self, pen_lambdas_W: list[float], pen_lambdas_H: list[float]):
        W = self.W
        Ht = self.Ht
        Vt = self.Vt

        if W.requires_grad:
            WHt = self.reconstruct(Ht.detach(), W)
            self._dbu_W(pen_lambdas_W, Vt, WHt, W)
            assert not torch.any(torch.isnan(W))

        if Ht.requires_grad:
            WHt = self.reconstruct(Ht, W.detach())
            self._dbu_H(pen_lambdas_H, Vt, WHt, Ht)
            assert not torch.any(torch.isnan(Ht))

    @torch.no_grad
    def loss(
        self, pen_lambdas_W: list[float], pen_lambdas_H: list[float]
    ) -> tuple[float, dict]:
        losses = {}
        losses["penalties_H"] = {}
        losses["penalties_W"] = {}
        full_loss = 0

        # T, K = H.shape
        # M, _ = V.shape  # M, K

        Ht = self.Ht.detach()
        Vt = self.Vt.detach()
        W = self.W.detach()
        WHt = self.reconstruct(Ht, W)

        full_loss = self.divergence.compute(Vt.T, WHt.T) / Vt.numel()
        assert not torch.any(torch.isnan(full_loss))
        full_loss = full_loss.item()
        losses["divergence"] = full_loss

        for penalty, lambda_ in zip(self.penalties_H, pen_lambdas_H):
            loss = lambda_ * penalty.compute(Ht) / Ht.numel()
            assert not torch.any(torch.isnan(loss))
            losses["penalties_H"][penalty.__class__.__name__] = loss.item()
            full_loss += loss.item()

        for penalty, lambda_ in zip(self.penalties_W, pen_lambdas_W):
            loss = lambda_ * penalty.compute(W) / W.numel()
            assert not torch.any(torch.isnan(loss))
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

    @H.setter
    def H(self, H: Tensor):
        assert not torch.any(torch.isnan(H))
        self.Ht.data.copy_(H.T)


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


class BetaDivergence(Divergence):
    def __init__(self, beta: float):
        self.beta = beta

    def compute(self, V: Tensor, Vhat: Tensor):
        V_eps = V.add(EPS)
        Vhat_eps = Vhat.add(EPS)
        if self.beta == 0:
            ret = V_eps / Vhat_eps - torch.log(V_eps / Vhat_eps) - 1
        elif self.beta == 1:
            ret = V_eps * (torch.log(V_eps) - torch.log(Vhat_eps)) + (Vhat_eps - V_eps)
        else:
            ret = (
                (
                    torch.pow(V_eps, self.beta)
                    + (self.beta - 1) * torch.pow(Vhat_eps, self.beta)
                    - self.beta * V_eps * torch.pow(Vhat_eps, self.beta - 1)
                )
                / self.beta
                / (self.beta - 1)
            )
        return torch.sum(ret)

    def grad_pos(self, V: Tensor, Vhat: Tensor):
        Vhat_eps = Vhat.add(EPS)
        return Vhat_eps.pow(self.beta - 1)

    def grad_neg(self, V: Tensor, Vhat: Tensor):
        V_eps = V.add(EPS)
        Vhat_eps = Vhat.add(EPS)
        return V_eps * Vhat_eps.pow(self.beta - 2)
    
    def __str__(self) -> str:
        return f"BetaDivergence(beta={self.beta})"


class L1(Penalty):
    """L1 regu, also known as Lasso"""

    def compute(self, X: Tensor):
        return X.sum().abs()

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        return torch.ones_like(X)


class L2(Penalty):
    """L1 regu, also known as Ridge"""

    def compute(self, X: Tensor):
        return X.pow(2).sum()

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        return 2 * X


class SmoothGain(Penalty):
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
        sub_x = X[:-1, :-1]
        sub_x_i_jp1 = X[:-1, 1:]
        sub_x_ip1_j = X[1:, :-1]
        sub_x_ip1_jp1 = X[1:, 1:]

        return (
            sub_x
            * (
                sub_x_i_jp1 * sub_x_ip1_jp1
                + sub_x_ip1_j * sub_x_ip1_jp1
                + sub_x_ip1_j * sub_x_i_jp1
            )
        ).sum()

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        ret = torch.zeros_like(X)

        X_ip1_jp1 = X[2:, 2:]  # shifted by +1 in both dims
        X_ip1_j = X[2:, 1:-1]  # shifted by +1 in the row dim
        X_i_jp1 = X[1:-1, 2:]  # shifted by +1 in the column dim
        X_im1_jp1 = X[:-2, 2:]  # shifted by -1 in the row dim, +1 in the column dim
        X_im1_j = X[:-2, 1:-1]  # shifted by -1 in the row dim
        X_i_jm1 = X[1:-1, :-2]  # shifted by -1 in the column dim
        X_ip1_jm1 = X[2:, :-2]  # shifted by +1 in the row dim, -1 in the column dim
        X_im1_jm1 = X[:-2, :-2]  # shifted by -1 in both dims

        ret[1:-1, 1:-1] = (
            X_i_jp1 * X_ip1_jp1
            + X_ip1_j * X_ip1_jp1
            + X_ip1_j * X_i_jp1
            + X_im1_j * X_i_jp1
            + X_im1_j * X_im1_jp1
            + X_i_jm1 * X_ip1_j
            + X_i_jm1 * X_ip1_jm1
            + X_im1_jm1 * X_im1_j
            + X_im1_jm1 * X_i_jm1
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


class NoHorizLines(Penalty):
    def compute(self, X: Tensor):
        return torch.sum(X[:, 1:] * X[:, :-1])

    def grad_neg(self, X: Tensor):
        return torch.zeros_like(X)

    def grad_pos(self, X: Tensor):
        ret = torch.zeros_like(X)
        ret[:, 1:-1] = X[:, 2:] + X[:, :-2]
        return ret
