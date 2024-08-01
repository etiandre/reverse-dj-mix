
== Gain smoothness

We expect $g(tau)$ to be varying smoothly over time. Thus we introduce the *gain smoothness* penalty, that minimizes the difference between two consecutive gain values.

Given that the gain of the column is given by 
$ g[tau] = sqrt(sum_(t=0)^(T-1) bold(H)_(t tau) ) $

$
g[tau]^2 - g[tau-1]^2 &= sum_(t=0)^(T-1) bold(H)_(t tau) - sum_(t=0)^(T-1) bold(H)_(t, tau-1) \
&= sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1)) 
$
So we define the penality function:

$
cal(P)_g (bold(H)) &= sum_(tau=1)^(K-1) sum_(t=0)^(T-1) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 \
$

*gradient calculation*
$
partial / (partial bold(H)_(i j)) (bold(H)_(t tau) - bold(H)_(t, tau-1))^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i,j+1) - bold(H)_(i j)) &"if" i=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
(partial cal(P)_g) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i, j-1)) -2(bold(H)_(i,j+1) - bold(H)_(i j)) \
 &= 4 bold(H)_(i j) - 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
$


*gradient term separation*:
$
gradient_bold(H)^+ cal(P)_g = 4 bold(H) \
(gradient_bold(H)^- cal(P)_g)_(i j) = 2 (bold(H)_(i,j-1) + bold(H)_(i,j+1))
$

```python
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
```


== Diagonal smoothness

We hypothesize the tracks to be played near their original speed, and that there will be significant time intervals without any loops or jumps. This appears in $bold(H)$ as diagonal line structures. We define a *diagonal smoothness* penalty that minimises the difference between diagonal cells of $bold(H)$:

$
cal(P)_d (bold(H)) = sum_(t=1)^(T-1) sum_(tau=1)^(K-1) (bold(H)_(t,tau) - bold(H)_(t-1, tau-1))^2
$

*gradient calculation*
$
partial / (partial bold(H)_(i j)) (bold(H)_(t tau) - bold(H)_(t-1, tau-1))^2 = cases(
  2(bold(H)_(i j) - bold(H)_(i-1, j-1)) &"if" i=t "and" j=tau,
  -2(bold(H)_(i+1,j+1) - bold(H)_(i j)) &"if" i+1=t "and" j+1=tau,
  0 &"otherwise"
)
$

So:
$
(partial cal(P)_d) / (partial bold(H)_(i j)) &= 2(bold(H)_(i j) - bold(H)_(i-1, j-1)) -2(bold(H)_(i+1,j+1) - bold(H)_(i j)) \
 &= 4 bold(H)_(i j) - 2 (bold(H)_(i-1,j-1) + bold(H)_(i+1,j+1))
$

*gradient term separation*:
$
gradient_bold(H)^+ cal(P)_d = 4 bold(H) \
(gradient_bold(H)^- cal(P)_d)_(i j) = 2 (bold(H)_(i-1,j-1) + bold(H)_(i+1,j+1))
$

```python
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
```

== Lineness

The time-remapping function is expected to be piecewise continuous. In $bold(H)$, this means we can characterize the neighboring cells of a given activation. Given an activated cell $(i,j)$, only the up direction $(i+1,j)$, right direction $(i,j+1)$, or upper-right diagonal direction $(i+1, j+1)$ should be activated, but not any combination of the three.

Thus we define the *lineness* penalty below, that gets larger when more than one of these direction are activated near an activated cell:

$
cal(P)_l (bold(H)) &= sum_(t=0)^(T-2) sum_(tau=0)^(K-2) bold(H)_(t,tau) (bold(H)_(t,tau+1) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t+1,tau+1) + bold(H)_(t+1,tau) bold(H)_(t,tau+1) )
$

*gradient calculation and separation*
$
(partial cal(P)_l) / (partial bold(H)_(i j))  =gradient_bold(H)^+ cal(P)_l &= bold(H)_(i,j+1) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i+1,j+1) + bold(H)_(i+1,j) bold(H)_(i,j+1) \
&+ bold(H)_(i-1,j) bold(H)_(i,j+1) + bold(H)_(i-1,j) bold(H)_(i-1,j+1) \
&+ bold(H)_(i,j-1) bold(H)_(i+1,j) + bold(H)_(i,j-1) bold(H)_(i+1,j-1) \
&+ bold(H)_(i-1,j-1) bold(H)_(i-1,j) + bold(H)_(i-1,j-1) bold(H)_(i,j-1) \
gradient_bold(H)^- cal(P)_l &= 0
$

```python
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
```