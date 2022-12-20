import numpy as np

from scipy.optimize import linear_sum_assignment


def perm_H(H_true: np.ndarray, H_pred: np.ndarray) -> np.ndarray:
    n = H_true.shape[1]
    # cost matrix
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = np.linalg.norm(H_pred[i] - H_true[j])

    # hungarian algorithm to get permutation of topic indices
    perm = sorted(linear_sum_assignment(J))

    # permute
    H_perm = np.zeros_like(H_pred)
    for i in range(n):
        H_perm[i] = H_pred[perm[i][1]]

    return H_perm


def perm_W(W_true: np.ndarray, W_pred: np.ndarray) -> np.ndarray:
    n = W_true.shape[0]
    # cost matrix
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = np.linalg.norm(W_pred[:, i] - W_true[:, j])

    # hungarian algorithm to get permutation of topic indices
    _, perm = linear_sum_assignment(J)

    # permute
    W_perm = np.zeros_like(W_pred)
    for i in range(n):
        W_perm[:, i] = W_pred[:, perm[i]]

    return W_perm
