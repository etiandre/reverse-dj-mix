import numpy as np
from typing import List, Callable, Optional
import warnings
from beta_nmf import BetaNMF
import scipy.ndimage
import scipy.signal
import logging

logger = logging.getLogger(__name__)
print(logger)

class ActivationLearner:
    def __init__(
        self,
        mix: np.ndarray,
        refs: List[np.ndarray],
        transform: Callable,
        inv_transform: Callable,
        additional_dim: int = 0,
        **nmf_kwargs,
    ):
        self.learn_add = additional_dim > 0
        self.transform = transform
        self.inv_transform = inv_transform
        self.refs = refs
        # transform audio into feature matrix
        refs_mat = [transform(i) for i in refs]
        mix_mat = transform(mix)

        # compute indexes of track boundaries
        self.split_idx = [0] + list(
            np.cumsum([ref.shape[1] for ref in refs_mat], axis=0)
        )

        # construct NMF matrices
        V = mix_mat / mix_mat.max()

        refs_mat = [i / i.max() for i in refs_mat]
        if self.learn_add:
            Wa = np.random.rand(refs_mat[0].shape[0], additional_dim)
            W = np.concatenate(refs_mat + [Wa], axis=1)
            self.split_idx.append(self.split_idx[-1] + additional_dim)
        else:
            W = np.concatenate(refs_mat, axis=1)

        # initialize activation matrix
        H = np.random.rand(W.shape[1], V.shape[1])

        logger.debug(f'Shape of W: {W.shape}')
        logger.debug(f'Shape of H: {H.shape}')
        logger.debug(f'Shape of V: {V.shape}')

        self.nmf = BetaNMF(V, W, H, 0, fixed_W=not self.learn_add, **nmf_kwargs)

    def iterate(self):
        if self.learn_add:
            # save everything except Wa
            W_save = self.nmf.W[:, : self.split_idx[-2]].copy()
            self.nmf.iterate()
            # copy it back
            self.nmf.W[:, : self.split_idx[-2]] = W_save
            # clip Wa
            self.nmf.W[:, self.split_idx[-2] : self.split_idx[-1]] = np.clip(
                self.nmf.W[:, self.split_idx[-2] : self.split_idx[-1]], 0, 1
            )
        else:
            self.nmf.iterate()
        self.nmf.H = np.clip(self.nmf.H, 0, 1)

        # Calculate loss
        loss = self.nmf.loss()
        assert not np.isnan(loss).any(), "NaN in loss"

        return loss

    def volume(self, weiner: Optional[int] = 3, medfilt: Optional[int] = 7):
        ret = []
        if weiner is not None:
            Hfilt = scipy.signal.wiener(self.nmf.H, mysize=(weiner, weiner))
        else:
            Hfilt = self.nmf.H
        sum = Hfilt.sum(axis=0)
        for left, right in zip(self.split_idx, self.split_idx[1:]):
            vol = Hfilt[left:right, :].sum(axis=0) / sum
            if medfilt is not None:
                vol = scipy.signal.medfilt(vol, medfilt)
            ret.append(vol)
        return ret

    def position(
        self, threshold=1e-2, weiner: Optional[int] = 3, medfilt: Optional[int] = 7
    ):
        ret = []
        if weiner is not None:
            Hfilt = scipy.signal.wiener(self.nmf.H, mysize=(weiner, weiner))
        else:
            Hfilt = self.nmf.H
        for left, right in zip(self.split_idx, self.split_idx[1:]):
            _, N = Hfilt.shape
            pos = np.empty(N)
            for i in range(N):
                col = Hfilt[left:right, i] ** 2
                if col.sum() >= threshold:
                    pos[i] = scipy.ndimage.center_of_mass(col)[0]
                else:
                    pos[i] = np.nan
            if medfilt is not None:
                pos = scipy.signal.medfilt(pos, medfilt)
            ret.append(pos)
        return ret

    def reconstruct(self, i: int):
        a = self.split_idx[i]
        b = self.split_idx[i + 1]
        if i < len(self.refs):
            ref_spec = self.transform(self.refs[i])
            Vi = (
                self.nmf.V * (ref_spec @ self.nmf.H[a:b, :]) / (self.nmf.W @ self.nmf.H)
            )
        else:  # no phase :(
            Vi = (
                self.nmf.V
                * (self.nmf.W[:, a:b] @ self.nmf.H[a:b, :])
                / (self.nmf.W @ self.nmf.H)
            )
            warnings.warn(f"Track {i} not in refs: i don't have phase info")
        return self.inv_transform(Vi)
