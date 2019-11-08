import numpy as np


def make_scoring_dict(alphabet, matrix):
    M = {}
    for i, a in enumerate(alphabet):
        M[a, None] = matrix[i][-1]
        M[None, a] = matrix[-1][i]
        for j, b in enumerate(alphabet):
            M[a, b] = matrix[i][j]
    return M


def banded_dp(S, k, s, t):
    m = len(s)
    n = len(t)
    W = 2 * k + 1
    # assume m >> n
    H = np.zeros((m+1, W+1))

    for j in range(1, k):
        pass
