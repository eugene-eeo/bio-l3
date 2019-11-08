import numpy as np


def make_scoring_dict(alphabet, matrix):
    M = {}
    for i, a in enumerate(alphabet):
        M[a, None] = matrix[i][-1]
        M[None, a] = matrix[-1][i]
        for j, b in enumerate(alphabet):
            M[a, b] = matrix[i][j]
    return M


def _dynprog(S, s, t):
    m = len(s)
    n = len(t)
    V = np.zeros((m+1, n+1), dtype=int)
    M = np.chararray((m+1, n+1))
    M[:] = ' '
    # Fill in vertical values
    for i in range(1, m+1):
        V[i, 0], M[i, 0] = max(
            (V[i-1, 0] + S[s[i-1], None], 'U'),
            (0, 'T'),
        )

    # Fill in horizontal values
    for j in range(1, n+1):
        V[0, j], M[0, j] = max(
            (V[0, j-1] + S[None, t[j-1]], 'L'),
            (0, 'T'),
        )

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_d = V[i-1, j-1] + S[s[i-1], t[j-1]]
            cost_u = V[i-1, j]   + S[s[i-1], None]
            cost_l = V[i, j-1]   + S[None, t[j-1]]
            V[i, j], M[i, j] = max(
                (cost_d, 'D'),
                (cost_u, 'U'),
                (cost_l, 'L'),
                (0,      'T'),
            )

    # Reconstruct
    i, j = np.unravel_index(V.argmax(), V.shape)
    max_score = V[i, j]
    s_idxs = []
    t_idxs = []

    while i != 0 or j != 0:
        c = M[i, j]
        if c == b'D':
            i -= 1
            j -= 1
            s_idxs.append(i)
            t_idxs.append(j)
        elif c == b'U': i -= 1
        elif c == b'L': j -= 1
        else: break

    s_idxs.reverse()
    t_idxs.reverse()
    return max_score, s_idxs, t_idxs


def dynprog(alphabet, scores, s, t):
    S = make_scoring_dict(alphabet, scores)
    return _dynprog(S, s, t)


# Dynamic programming with O(n) space


def score_global_alignment(S, s, t):
    m = len(s)
    n = len(t)
    V = np.zeros((2, n+1))

    for j in range(1, n+1):
        V[0, j] = V[0, j-1] + S[None, t[j-1]]

    for i in range(1, m+1):
        V[1, 0] = V[0, 0] + S[s[i-1], None]
        for j in range(1, n+1):
            V[1, j] = max(
                V[0, j-1] + S[s[i-1], t[j-1]],
                V[0, j]   + S[s[i-1], None],
                V[1, j-1] + S[None, t[j-1]],
            )
        V[0, :] = V[1, :]
    return V[0]


def global_align(S, X, Y):
    Z = []
    W = []
    s = 0
    if len(X) == 0:
        s = sum(S[None, y] for y in Y)
    elif len(Y) == 0:
        s = sum(S[x, None] for x in X)
    elif len(X) == 1 or len(Y) == 1:
        return _dynprog(S, X, Y)
    else:
        xlen = len(X)
        xmid = xlen // 2
        L = score_global_alignment(S, X[:xmid], Y)
        R = score_global_alignment(S, X[xmid:][::-1], Y[::-1])
        ymid = (L + R[::-1]).argmax()

        s1, Z1, W1 = global_align(S, X[:xmid], Y[:ymid])
        s2, Z2, W2 = global_align(S, X[xmid:], Y[ymid:])

        Z = Z1 + [z + xmid for z in Z2]
        W = W1 + [w + ymid for w in W2]
        s = s1 + s2
    return s, Z, W


def find_local_max(S, s, t):
    m = len(s) + 1
    n = len(t) + 1
    V = [[(0, (0, 0))] * n for _ in range(2)]
    # (score, start, end)
    best = (0, (0, 0), (0, 0))

    for j in range(1, n):
        V[0][j] = max(
            (V[0][j-1][0] + S[None, t[j-1]], V[0][j-1][1]),
            (0, (0, j)),
        )
        best = max(best, (*V[0][j], (0, j)))

    for i in range(1, m):
        V[1][0] = max(
            (V[0][0][0] + S[s[i-1], None], V[0][0][1]),
            (0, (i, 0)),
        )
        best = max(best, (*V[1][0], (i, 0)))

        for j in range(1, n):
            V[1][j] = max(
                (V[0][j-1][0] + S[s[i-1], t[j-1]], V[0][j-1][1]),
                (V[0][j][0]   + S[s[i-1], None],   V[0][j][1]),
                (V[1][j-1][0] + S[None, t[j-1]],   V[1][j-1][1]),
                (0, (i, j)),
            )
            best = max(best, (*V[1][j], (i, j)))
        V[0], V[1] = V[1], V[0]
    return best[1], best[2]


def dynproglin(alphabet, scores, s, t):
    S = make_scoring_dict(alphabet, scores)
    (si, sj), (ei, ej) = find_local_max(S, s, t)
    # If one of the substrings is empty, then just give up now.
    if ei == si or ej == sj:
        return 0, [], []
    s, Z, W = global_align(S, s[si:ei], t[sj:ej])
    return s, [z + si for z in Z], [w + sj for w in W]


# Heuristic Method (FASTA-lite)


def banded_dp(S, k, s, t):
    m = len(s)
    n = len(t)
    U = k
    L = -k
    W = U - L + 1

    if W > m or W > n:
        return _dynprog(S, s, t)

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


def compute_index_table(ktup, s):
    index_table = {}
    for i in range(len(s) - ktup + 1):
        sub = s[i:i+ktup]
        if sub not in index_table:
            index_table[sub] = [i]
        else:
            index_table[sub].append(i)
    return index_table


def find_seeds(S, ktup, index_table, t):
    pass
