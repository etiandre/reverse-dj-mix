import sparse
import numpy as np

# SparseType = sparse.COO
SparseType = np.ndarray
ArrayType = np.ndarray | SparseType


def sparse_to_dense(x):
    if isinstance(x, sparse.SparseArray):
        return x.todense()
    return x


def dense_to_sparse(x):
    if isinstance(x, np.ndarray) and SparseType is not np.ndarray:
        return SparseType.from_numpy(x)
    return x
