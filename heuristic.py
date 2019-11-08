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
    U = k
    L = -k
    W = U - L + 1

    if W > m or W > n:
        print("NO U")
        return

    H = np.zeros((m+1, W+2), dtype=np.int32)
    E = np.zeros((m+1, W+2), dtype=np.int32)
    F = np.zeros((m+1, W+2), dtype=np.int32)
    M = np.chararray((m+1, W+2))
    M[:] = ' '

    lo_diag = 1 - L
    hi_diag = U - L + 1
    lo_row = 0
    hi_row = min(m, n - L)

    ld = lo_diag
    hd = hi_diag
    for i in range(lo_row+1, hi_row+1):
        # lld = max(1, lo_diag - i)
        # hhd = hi_diag if i <= n - U else hi_diag - (i - (n - U))
        if ld > 1: ld -= 1
        if i > n - U: hd -= 1

        for j in range(ld, hd + 1):
            rj = (j + L - 1 + i) - 1
            if rj < 0:
                continue
            F[i, j] = max(F[i, j-1], H[i, j-1] + S[s[i-1], None])
            E[i, j] = max(E[i-1, j+1], H[i-1, j+1] + S[None, t[rj]])
            H[i, j], M[i, j] = max(
                (H[i-1, j] + S[s[i-1], t[rj]], 'D'),
                (E[i, j], 'L'),
                (F[i, j], 'U'),
                (0, 'T'),
            )

    i, j = np.unravel_index(H.argmax(), H.shape)
    max_score = H[i, j]
    s_idxs = []
    t_idxs = []

    while i != 0 or j != 0:
        # lld = max(1, lo_diag - i)
        # hhd = hi_diag if i <= n - U else hi_diag - (i - (n - U))
        c = M[i, j]
        if c == b'D':
            s_idxs.append(i - 1)
            t_idxs.append(j + L - 1 + i - 1)
            i -= 1
        elif c == b'L':
            j -= 1
        elif c == b'U':
            i -= 1
        else:
            break

    s_idxs.reverse()
    t_idxs.reverse()
    return max_score, s_idxs, t_idxs
